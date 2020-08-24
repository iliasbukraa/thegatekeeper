# Gatekeeper

Currently, every country is being challenged by COVID19. This means that everywhere in the world people should be wearing face masks for each others safety. 
Wearing a face mask is still relatively new for a lot of people around the globe. Also unfortunately not everyone does abide to the law, the rules and best 
practices on how to wear a face mask. This is where the gatekeeper comes in.

## running Gatekeeper

By following the instructions in **requirements.txt** a virtual environment can be created to run the gatekeeper. A pretrained model is available.
To run the gatekeeper on recorded video, execute following command:
```
python video_upload.py [VIDEOPATH] [OUTPUTPATH]
```

To run the gatekeeper on a live webcam feed, execute following command:
```
python video_live_modeltrained.py
```

If the user wishes to train their own model, do so by executing following command:
```
python data_training.py
```

![](Gatekeeper-Logo.png)
