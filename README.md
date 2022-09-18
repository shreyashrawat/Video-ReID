# Video-ReID

Implementation of Deep-person-ReID

### Findings

I tried implementing 3 libraries for the video implementation of person Re-id:

1. https://github.com/Mhttx2016/Multi-Camera-Object-Tracking-via-Transferring-Representation-to-Top-View - This was the simplest looking library and was very simple to use. The downside of this library was that the detection file was already present so there were not using a detector to get the detections first instead using the pre-used detections. Second, this library had a lot of library compatability issues. Some code was written in Tf 2.0 while some in Tf 1.0, converting that whole code into a consistent version was a very time consuming procedure.

2. https://github.com/layumi/Person_reID_baseline_pytorch - This was a comparatively well documented and supported library. It also had a GPU version implementation (which I thought would be a life saver) but the library was primarily built only for images. The performance was equal to state of the art on the Market1501 dataset but had to be finetuned and chaged a bit for different datasets.

3. https://github.com/KaiyangZhou/deep-person-reid - I finally went ahead with deep-person-reid due to various reasons. First, this had pre-built support for image and video data managers. No need to create your custom loaders for benchmark datasets. Second, it had a lot of flexibility in the model selection, cross training and testing (directly training on one dataset and testing on another). It also had a lot of datasets that automatically get downloaded and prepared for you although this was not present for video datasets. 

I went ahead with a resnet 50 pretrained model mainly due to it's past performances on re-id datasets. In spite of the model being a bit bulky, the results outweigh the drawbacks which led me to chosse this model. I fine tuned the resnet 50 model on the MARS Video Re-id dataset and here is what the statistics of the dataset looks like.

![image](https://user-images.githubusercontent.com/38159161/190922183-0bc505ef-26e4-43dd-89af-d18e50b8bfb9.png)

I finetuned the model on 20 epochs on the mars dataset with train_batch size as 3 and sequence_len as 15. The following is an image for training in progress. The training took about 5-6 hours due to absence of fast GPUs.

![image](https://user-images.githubusercontent.com/38159161/190922644-b327a503-a1a4-4369-9c58-7633df409e96.png)

So the results on a 3rd dataset produced these results where rank@1 which is essentially for the model was around 62% and mAP (mean average precision) was around 60%. Due to the crunch in time, I trained the model only on 20 epochs that led to average results which could have been improved with more experimentation and epochs.

## Implementation

The run_reid is the code used to train and test the model on the mars dataset after the deep-person-reid environment has been setup. Make sure the conda env is created and active after which the python script will do the training and testing.
