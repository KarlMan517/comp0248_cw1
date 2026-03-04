# Object Detection Coursework 1

This repo included all code used in the object detection coursework 1


## Instruction
```
cd 25067520_Man

conda create -n comp0248_cw1 python=3.10
conda activate comp0248_cw1

pip install -r requirements.txt
```

After setup the environment, you can run the training, evaluation, and visualization code.


## Training
To train the model, run **train.py** under the **src** directory.

## Evaluation
To get the metrics and confusion matrix, run **evaluation.py** under the **src** directory. The **confusion matrix** plot will be saved under the **results** directory, and the metrics will be printed in the console.
Remember to correct the test dataset path and also the checkpoint path (Which default saved under the **weights** directory)

## Visualization
To visualized the results, run **visualise.py** under the **src** directory. The **visualized results** will be saved under the **results** directory.
Remember to correct the test dataset path and also the checkpoint path (Which default saved under the **weights** directory)
The **num_samples** can be changed to show different numbers in the plot



