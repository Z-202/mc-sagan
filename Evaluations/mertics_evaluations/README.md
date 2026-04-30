# Evaluation Metrics

This directory provides the evaluation script used to compute **3D MSE**, **PSNR**, and **SSIM** between the generated MRI volumes and their corresponding ground-truth targets.

## Prerequisite

Before running the evaluation, you must first **train the 3D-MC-SAGAN model** and generate the synthetic MRI volumes.  
The evaluation script expects the generated outputs to be saved under the `generated_multi_proc` directory.

## Run the evaluation

Use the following command:

```bash
python eval_metrics.py --root /dir/to/generated_multi_proc --device cuda:0
