from unet.archive.data import *
from model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


data_gen_args = dict(
    rotation_range=0.2,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
    fill_mode='nearest')
# myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)
myGene = trainGenerator(
    batch_size=2,
    train_path='data/brats/train',
    image_folder='image',
    mask_folder='label',
    aug_dict=data_gen_args,
    save_to_dir='data/brats/train/aug')

model = unet()
model_checkpoint = ModelCheckpoint('unet_brats.keras', monitor='loss', verbose=1, save_best_only=True)
model.fit(myGene, steps_per_epoch=300, epochs=1, callbacks=[model_checkpoint])

testGene = testGenerator_brats("data/brats/test", num_image_from=88, num_image_to=108)
results = model.predict(testGene, 30, verbose=1)
saveResult_v2("data/brats/test", results)
