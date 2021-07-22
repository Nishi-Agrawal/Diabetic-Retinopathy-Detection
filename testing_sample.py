
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator



def prediction():
    

    
    
    mod=load_model('model.hd5')
    
    test_gen = ImageDataGenerator(rescale = 1./255)

    import os
    # PROJECT_PATH = os.path.abspath(os.path.dirname('uploaded.jpg'))
    # CAPTHA_ROOT = os.path.join(PROJECT_PATH,r'test_images\uploaded')
    #path=r'test_images\uploaded'
    test_data = test_gen.flow_from_directory(r'test_images/',
                                              target_size = (64, 64),
                                              batch_size = 32,
                                              class_mode = 'binary', shuffle=False)

    predicted = mod.predict_generator(test_data)
    print(predicted)
    y_pred = predicted[0][0] > 0.4
    percent_chance = round(predicted[0][0]*100, 2)
    print(percent_chance)

    return y_pred, percent_chance
# In[17]:

if __name__ == '__main__':
    print(prediction())


