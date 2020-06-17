# Show N Tell: Progress Update

**Team: Sunint B. &amp; David C.**

This sub-directory consists of code for Keras-implementation of Mask RCNN.

_Click on any script to view details._

## Procedure

To satisfy requirement for using GPUs, we have abstracted our Masked RCNN as separate _.py_ scripts, each consisting of a `main()` procedure:

**Phase 1.** `train.py`

* Exports weights as _.h5_
* Saves _Keras history_ object to _.pkl_

**Phase 2.** `predict.py`

* Saves results to CSV file
* Currently, only the model for _ClassId_ were fully developed

Pre-defined custom classes &amp; templates, as well as reusable objects, are implemented in `classes.py` to be loaded within either or both steps,

To run either script, simply run the command (written in lines 34-35 of `queue.pbs`)

```shell script
python3 train.py
python3 predict.py
```

(Note that only _Python3.x_ is supported.)

## Note on GPU usage

We made use of Dartmouth Research Computing GPU. Specific details as to the CPU &amp; GPU server is specified in the `PBS` script

```shell script
## Create virtual environment (do only ONCE)
## Virtual environment activated in queue.pbs
conda create --name kaggle

## Submit job (lines 34-35 most important)
qsub queue.pbs
```

### Limitations

* Currently, only single GPU is supported. Multi-GPU usage is unsupported.

## Data Exploration &amp; Visualization

We used _Jupyter Notebook_ for interactively making figures and plots for the presentation. Simply click on the following _.ipynb_ files to see the assets:

* [visualization_image_eda.ipynb](https://gitlab.cs.dartmouth.edu/ydavidchen/cs69-imaterialist-2020/blob/master/show-n-tell/visualization_image_eda.ipynb)
* [visualization_training_results.ipynb](https://gitlab.cs.dartmouth.edu/ydavidchen/cs69-imaterialist-2020/blob/master/show-n-tell/visualization_training_results.ipynb)

## References

* Shapes starter code / tutorial
* Mask RCNN starter code / notebook on Kaggle
* Dartmouth Research Computing GPU &amp; Linux (CPU) tutorial
