# Word-correction-using-Deep-Learning

Author: Ankit Raj

1. Prerequisites of this project

 -> Libraries required:
    -Keras
    -numpy
    -editdistance
    -random
    -os
    -pickle
    -h5py
    -TensorFlow(keras uses TensorFlow at backend)
     
   
     
 -> Dataset
     google's 1 billions words corpus
     "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz"
    
 

2. Instruction for Testing the Code:
   (On Terminal)
  -> run Project_File/main.py file
     -Input incorrect words and press Enter
  





3. Approach, bottlenecks and things to do
  -> Create 'data' folder containing 'unprocessed' and 'processed' folder.
  -> store google's 1 billion corpus in the 'data/unprocessed' folder.
  -> Preparation of data
     introducing spelling errors to a given sentence
     four types of spelling mistakes
     type = 0 --> replace one character
     type = 1 --> remove one character
     type = 2 --> insert one character
     type = 3 --> interchance positions of two next to next characters
     and store these sentences  with correct sentence in data/processed folder for training of the model(no pretrained weights are used)
   
 -> Choosing model
     LSTM(Long short Term Memory) is used in this assignment.keras framework is used for the implementation of the model.
     Encoder and Decoder architecture approach is used in this assignment.
     optimizer used is adam which provide higher accuracy in less time as compare to other optimizers.


 
     
  



