from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
import numpy as np
import matplotlib.pyplot as plt

model = InceptionResNetV2(weights='imagenet')
count = 0
for i in range(60):
    img_path = 'crop_image/bike_'+str(i)+'.jpg'
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
    t3 = decode_predictions(preds, top=3)[0]
    found = False
    for result in t3:
        
        if result[1] == 'motor_scooter':
            # plt.imshow(img)
            # plt.show()
            count+=1
            print(i, 'is motor scooter')
            found = True
    if not found:
        print(i, 'is something else', t3)
print('scooter found',count)

# Predicted: [(u'n02504013', u'Indian_elephant', 0.82658225), (u'n01871265', u'tusker', 0.1122357), (u'n02504458', u'African_elephant', 0.061040461)]