#!/usr/bin/env python3

import argparse
from functools import partial
import os
import shutil
import sys

import subprocess as subp

import SimpleITK as sitk
import support

print = partial(print, flush=True)

PREFIX_BASE = "/var/niftynet"
PREFIX = os.path.join(PREFIX_BASE, "work")

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-volume", required=True, help="Input volume to segment")
parser.add_argument("-o", "--output-label", required=True, help="Segmented output file")
parser.add_argument("-r", "--resampled-output", required=False, help="Resampled output file, if desired")
parser.add_argument("-g", "--gpus", type=int, default=1, help="Number of GPUs to use")
parser.add_argument("--cpu-resampling", action="store_true", help="Disable CUDA resampling and use the CPU instead")
parser.add_argument("--cuda", default="", help="Comma-separated IDs of CUDA devices to use (defaults to all)")
parser.add_argument("-t", "--threads", type=int, default=4, help="Number of threads to use")
parser.add_argument("--no-allow-growth", action="store_true", help="Disable TensorFlow GPU memory growth (may cause out-of-memory errors on systems with low VRAM)")

args = parser.parse_args()

if not args.no_allow_growth:
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

## Be very careful with adding quotes to "string" parameters in this template, e.g. if you put 
## quotes around the `csv_file` path then NiftyNet ignores it
config_template = """############################ input configuration sections
[ct]
csv_file = {prefix}/run.csv
spatial_window_size = (144, 144, 144)
interp_order = 1
axcodes = (A, R, S)

############################## system configuration sections
[SYSTEM]
cuda_devices = "{cuda}"
num_threads = {threads}
num_gpus = {gpus}
model_dir = /var/niftynet/models/Temporal_Bone_segmentation 
queue_length = 60

[NETWORK]
name = dense_vnet
batch_size = 10
histogram_ref_file = /var/niftynet/models/Temporal_Bone_segmentation/standardisation_models.txt
whitening = True
normalisation = True
normalise_foreground_only=True
foreground_type = mean_plus
cutoff = (0.001, 0.999)

[INFERENCE]
border = (36, 36, 36)
inference_iter = 20200
output_interp_order = 0
spatial_window_size = (144, 144, 144)
save_seg_dir = {prefix_base}/output

############################ custom configuration sections
[SEGMENTATION]
image = ct
label_normalisation = False
output_prob = False
num_classes = 10
"""

#config_file = os.path.join(PREFIX, "extensions/Temporal_Bone_segmentation/config_inference.ini")
config_file = os.path.join(PREFIX, "config_inference.ini")
## Write the configuration file for this run. This just makes sure that the config is using the 
## correct parameters as given by the user (e.g. GPU count). The input/output files are hardcoded
## into the template: we have to convert the input and output to and from the NIfTI format anyways
## and just put the results in the correct places
with open(config_file, 'w') as f:
    f.write(config_template.format(
        gpus=args.gpus, 
        threads=args.threads, 
        cuda=args.cuda, 
        prefix=PREFIX, 
        prefix_base=PREFIX_BASE
    ))

## Write the CSV
with open(os.path.join(PREFIX, "run.csv"), 'w') as f:
    f.write("run,{prefix}/run.nii\n".format(prefix=PREFIX))

## Now we have to convert the input volume into the appropriate format for the network
print("Converting input file...")
inp = sitk.ReadImage(args.input_volume)
print("Resampling...")
inp = support.fix_spacing(inp, use_sitk=args.cpu_resampling)
sitk.WriteImage(inp, os.path.join(PREFIX, "run.nii"))
if args.resampled_output is not None:
    print("Storing resampled image")
    sitk.WriteImage(inp, args.resampled_output)

## Run the actual segmentation
print("Calling net_segment...")
subp.check_call(["net_segment", "inference", "-c", config_file])

## Read the inferred filename from the outputted CSV
inferred = dict((i.strip("\r\n\t ").split(',') for i in open(os.path.join(PREFIX_BASE, "output/inferred.csv"), 'r').read().split("\n") if i.strip("\r\n\t ")))
## Convert the output image
outext = inferred["run"][::-1].split('.', 1)[0][::-1] if '.' in inferred["run"] else "nii.gz"
print("Loading output segmentation...")
outp = sitk.ReadImage(inferred["run"])
print("Running island removal...")
outp = support.island_removal(outp)
print("Storing output file...")
sitk.WriteImage(outp, args.output_label)

## Set the right permissions on the output image. We set them to the most permissive, just in case 
## the user isn't running the container as their own user (DeepInfer would otherwise be unable to 
## read/delete the output)
try:
    os.chmod(args.output_label, 0o777)
except:
    print("Couldn't set permissions on output file")

print("All done!")
