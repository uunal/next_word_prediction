# Next word prediction
Simple application using transformers models to predict next word or a masked word in a sentence.

The purpose is to demo and compare models in a fast fashion.

### Application
This app implements two variants of the same task (predict <mask> token). The first one consider the <mask> is at end of the sentence, simulating a prediction of the next word of the sentece.

The second variant is necessary to include a <mask> token where you want the model to predict the word.

![Word prediction](word_prediction.gif)

### Running 
```
cd app-folder
python app.py
```
Open your browser http://localhost:8000

### Important!
Currently trained models are not public



