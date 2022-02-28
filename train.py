import logging
import os
import time

import floorplans
from segmentation_models import xnet
from tensorflow.keras.optimizers import Adam
from tensorflow import losses, metrics

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)


def main():
    LEARNING_RATE = 1e-4

    model = xnet.Xnet(backbone_name='resnet50', classes=3)

    model.compile(loss=losses.SparseCategoricalCrossentropy(),
                  optimizer=Adam(learning_rate=LEARNING_RATE),
                  metrics=[metrics.SparseCategoricalAccuracy()],
                  )

    train_dataset, validation_dataset = floorplans.load_data()

    model.fit(train_dataset,
              validation_dataset,
              epochs=2,
              batch_size=1,
              verbose=2)

    model.save("unet_pp")


if __name__ == "__main__":
    tic = time.time()
    main()
    toc = time.time()
    print('total training + evaluation time = {} minutes'.format((toc - tic) / 60))
