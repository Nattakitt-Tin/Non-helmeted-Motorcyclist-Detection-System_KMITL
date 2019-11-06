from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.models import Model,Sequential,load_model
from keras.layers import Dense, GlobalAveragePooling2D,Flatten,Dropout
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras import backend as K
from google.colab import drive
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

drive.mount('gdrive')
dir = 'gdrive/My Drive/Colab Notebooks/Helmet_Detection/'
history_dict = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': [] }

def build_model(freeze_split = 520):
  print('Building new model. . .')
  # base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299,299,3))
  base_model = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(299,299,3))
  # base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
  print('Base model loaded')

  # for layer in base_model.layers:
  #   layer.trainable = Tr

  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  # x = Dense(1000, activation='relu')(x) #input = (8,8,2048)
  # x = Dropout(rate = 0.3)(x)
  predictions = Dense(2, activation='softmax')(x)
  model = Model(inputs=base_model.input, outputs=predictions)
  
  # for layer in model.layers[:freeze_split]:
  #   layer.trainable = False
  # for layer in model.layers[freeze_split:]:
  #   layer.trainable = True


  model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy']) #compile 'after' freeze
  # model.summary()
  return model

def k_fold(generator, n_splits, BATCH_SIZE):
  ''' 
    # return: a generator of dict {'train','test','steps_per_fold'} 
  '''
  steps_per_fold = generator.samples // BATCH_SIZE // n_splits
  step = 0
  def generate(fold_type, test_number):
    nonlocal step
    while True:
      np.random.seed(93)
      if step+1 > steps_per_fold*n_splits:
        step = 0
      step+=1
      is_test = (step-1)//steps_per_fold is test_number
      batch = next(generator)
      # if step%steps_per_fold == 1:
      #   print()
      #   print('step',step,fold_type,'is test',is_test)
      #   plt.imshow(batch[0][0])
      #   plt.show()
      if fold_type is 'train' and not is_test:
        # print('step', step, fold_type)
        yield batch
      elif fold_type is 'test' and is_test:
        # print('step', step, fold_type)
        yield batch
      # else:
        # print('step', step, '<skip>',fold_type , steps_per_fold, test_number)

  for n in range(n_splits):
    print('test on fold '+str(n+1)+'/'+str(n_splits))
    yield {'train': generate('train', n), 'test': generate('test', n), 'steps_per_fold':steps_per_fold}


def update_history(history):
  history_dict['acc'] += history.history['acc']
  history_dict['val_acc'] += history.history['val_acc']
  history_dict['loss'] += history.history['loss']
  history_dict['val_loss'] += history.history['val_loss']

def plot_training(min = -1):
  epochs = range(len(history_dict['acc']))
  plt.plot(epochs, history_dict['acc'], 'g--',)
  plt.plot(epochs, history_dict['val_acc'], 'g-')
  if 0 < min < 1:
    plt.ylim(min, 1.) 
  plt.title('Training and validation accuracy')
  # plt.figure()
  # plt.plot(epochs, history_dict['loss'], 'r--')
  # plt.plot(epochs, history_dict['val_loss'], 'r-')
  # plt.title('Training and validation loss')
#       plt.ylim(0., 0.5) 
  plt.show()
#     plt.savefig('gdrive/My Drive/Colab Notebooks/acc_vs_epochs.png')

print('done')
# K-folds cross validation
BATCH_SIZE = 30
EPOCH = 10
n_split = 5
folds_score = []
INPUT_SHAPE = (299,299)

IDG = ImageDataGenerator(rescale=1./255)
test_generator = IDG.flow_from_directory(dir+'dataset/test', class_mode='categorical',target_size=INPUT_SHAPE, batch_size=BATCH_SIZE, shuffle=True)
data_generator = IDG.flow_from_directory(dir+'dataset/train',class_mode='categorical',target_size=INPUT_SHAPE, batch_size=BATCH_SIZE, shuffle=True)

for data in k_fold(data_generator, n_split, BATCH_SIZE):
  history_dict = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': [] }
  model = build_model()
  
  history = model.fit_generator(data['train'], 
                                epochs = EPOCH, workers = 1,
                                steps_per_epoch = data['steps_per_fold'] * (n_split-1),
                                validation_data = data['test'],
                                validation_steps = data['steps_per_fold'])
  update_history(history) # print each epoch
  plot_training(0.5)
  # epochs end here


  ev = model.evaluate_generator(test_generator)
  folds_score.append(ev[1])
  print('folds acc',folds_score)
  print('# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #')
print("Average Accuracy from "+str(n_split)+" split "+str(EPOCH)+ "epochs : "+ str(sum(folds_score)/len(folds_score)) )