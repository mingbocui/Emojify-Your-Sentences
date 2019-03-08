# Emojify-Your-Sentences❤️❤️❤️❤️

#### This repository provides two ways of adding emoji to your sentences, just like:  
"I love you ❤️"  
"I am so sad 😞"  
### How to start?  
just clone every thing to your local computer and *unzip* the glove.txt in the data folder.   
```python  
For Model 1: python -m Emojify-V1.py    
For Model 2: python -m Emojify-V2.py  
```

## Model 1: Word Vector Representation
Based on the Glove word2vector map we can represent every word with a vector, then add every word-vector appeared in the sentence followed by a softmax we could get a average vector which represent this sentence. Using linear regression which train Weight and bias based on gradient descent we could get a 5-dimensional output(Because we only have 5 emojis). Finding the index which the max value resides in this vector could lead us to find the correct emojis.  
Word-Vector representation is powerful, but it suffers from **inability** to memory the context of words, for example, if you input "I feel not very happy", the model will give your the output of:  
"I feel not very happy 😄"  
<img src="https://github.com/mingbocui/Emojify-Your-Sentences/blob/master/images/data_set.png" style="width:100px;height:50;">  
![work process](https://github.com/mingbocui/Emojify-Your-Sentences/blob/master/images/image_1.png)

## Model 2: LSTM comes to the rescue
As we know that LSTM is pretty good at memory the contexts, so applying LSTM to this task may be some of help.  
Differ from the method 1, we use another method to represent the word in a sentence:  
Every word will be changed to a number wich represent the index in the word map. Because the LSTM need we input the sequence with same length, so we have to embedding the short sentences with zeros.  
![embedding layer](https://github.com/mingbocui/Emojify-Your-Sentences/blob/master/images/embedding1.png)
The LSTM network we will use:(deeper layer will *not* help but harm the performance):  
![LSTM](https://github.com/mingbocui/Emojify-Your-Sentences/blob/master/images/emojifier-v2.png)
