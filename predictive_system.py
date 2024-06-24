import numpy as np
import joblib

# loading the saved model
loaded_model = joblib.load(open('logreg_model.pkl', 'rb'))


input_data = ('test')

# changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0] == 0):
  print('The news is fake')
else:
  print('The news is not fake')