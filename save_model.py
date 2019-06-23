import os
import model_definition

model_path = os.path.join('model', 'xception_v5_1000_acc.h5')

model = model_definition.load_xception()
model.save(model_path)
