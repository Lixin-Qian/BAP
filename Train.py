# basic library
import os
# import tensorflow
import tensorflow.compat.v1 as tf
# import keras
from tensorflow import keras
# import cleverhans
from cleverhans.utils_keras import KerasModelWrapper
# import pruning
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_callbacks import UpdatePruningStep, PruningSummaries
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_scope
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PolynomialDecay
from tensorflow_model_optimization.python.core.sparsity.keras.prune import prune_low_magnitude
from tensorflow_model_optimization.python.core.sparsity.keras.prune import strip_pruning
# import my library
from Adversarial import Adversarial_Model
from Adversarial_loss import get_adversarial_acc_metric, get_adversarial_loss, get_DATAA_acc_metric, get_DATAA_loss


class Train:
    def __init__(self, sess, model, dataset,
                 nb_epochs, learning_rate = 1e-3, batch_size = 128,
                 load = None, save = 'model.h5',
                 attack = None, adv_train = None, eps = None, iterations = None,
                 prun_para = None):
        # TO DO: Para set into Para.py
        # tensorboard file
        self.logdir = './logs/'
        self.model = model

        if attack is not None:
            print('#### Attack Model with {} ####'.format(attack))
            wrap_model = KerasModelWrapper(self.model)
            # TO DO: need doing
            at_model = Adversarial_Model(adv = attack, eps = eps, iterations = iterations).adv_model
            attack_params = Adversarial_Model(adv = attack, eps= eps, iterations = iterations).adv_params
            attack_model = at_model(wrap_model, sess = sess)



        if prun_para is not None:
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                str(prun_para['final_sparsity']) + '_' + str(
                    adv_train) + '_{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.h5',
                monitor='val_acc', verbose=0, save_best_only=False)
            model_checkpoint2 = keras.callbacks.ModelCheckpoint(
                str(prun_para['final_sparsity']) + '_' + str(
                    adv_train) + '_{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.h5',
                monitor='val_loss', verbose=0, save_best_only=False)
            print('#### Prune the Model with{} ####'.format(prun_para))
            self.keras_file = str(prun_para['final_sparsity']) + '_' + str(nb_epochs) + '_' + str(save)
            # Add a pruning step callback to peg the pruning step to the optimizer's
            # step. Also add a callback to add pruning summaries to tensorboard
            callbacks = [
                # Update the pruning step
                UpdatePruningStep(),
                # Add summaries to keep track of the sparsity in different layers in training
                PruningSummaries(log_dir= self.logdir, profile_batch=0),
                # tensorboard
                # keras.callbacks.TensorBoard(log_dir= self.logdir, profile_batch=0),
                # earlystopping
                # keras.callbacks.EarlyStopping(monitor='val_loss', patience= 1, verbose= 0, mode='auto'),
                # 不能用 ModelCheckpoint
                model_checkpoint,
                model_checkpoint2
            ]
        else:
            model_checkpoint = keras.callbacks.ModelCheckpoint(
                str(adv_train) + '_{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.h5',
                monitor='val_acc', verbose=0, save_best_only=False)
            model_checkpoint2 = keras.callbacks.ModelCheckpoint(
                str(adv_train) + '_{epoch:02d}-{val_acc:.4f}-{val_loss:.4f}.h5',
                monitor='val_loss', verbose=0, save_best_only=False)
            self.keras_file = str(nb_epochs) + '_' + str(save)
            # Load TensorBoard to monitor the training process
            callbacks = [
                # keras.callbacks.TensorBoard(log_dir= self.logdir, profile_batch=0),
                # keras.callbacks.EarlyStopping(monitor='val_loss', patience= 1, verbose=0, mode='auto'),
                model_checkpoint,
                model_checkpoint2
            ]

        # TO DO: class save
        if load is not None:
            print('#### Load Model from:{} ####'.format(load))
            if prun_para is not None:
                with prune_scope():
                    self.model = keras.models.load_model(load)
            else:
                self.model = keras.models.load_model(load)

        # set optimizers
        opt = keras.optimizers.Adam(lr = learning_rate)
        # opt = keras.optimizers.SGD(lr = learning_rate)

        # set training loss
        if adv_train is not None:
            # adversarial training
            if adv_train != attack and adv_train != 'bat':
                # adversarial model
                w_model = KerasModelWrapper(self.model)
                adv_model = Adversarial_Model(adv= adv_train, eps= eps, iterations = iterations).adv_model
                adv_params = Adversarial_Model(adv= adv_train, eps= eps, iterations = iterations).adv_params
                adversarial_model = adv_model(w_model, sess=sess)
                loss = get_adversarial_loss(self.model, adversarial_model, adv_params)
            elif adv_train == attack and adv_train != 'bat':
                adversarial_model = attack_model
                adv_params = attack_params
                loss = get_adversarial_loss(self.model, adversarial_model, adv_params)
            elif adv_train == 'bat':
                # adversarial in BAT
                w_model = KerasModelWrapper(self.model)
                adv_model = Adversarial_Model(adv= 'deepfool', iterations=iterations).adv_model
                adv_params = Adversarial_Model(adv= 'deepfool', iterations=iterations).adv_params
                adversarial_model = adv_model(w_model, sess=sess)
                loss = get_DATAA_loss(self.model, adversarial_model, adv_params)
            print('#### Adversarial Training in {} ####'.format(str(adv_train)))

        else:
            # normal training
            print('#### Normal Training ####')
            loss = keras.losses.categorical_crossentropy

        # set metrics
        if attack is not None:
            print('#### Adv Acc & Clean Acc ####')
            adv_acc_metric = get_adversarial_acc_metric(self.model, attack_model, attack_params)
            metrics= ['accuracy', adv_acc_metric]
        else:
            # if adv_train == 'bat':
                # adv_acc_metric = get_adversarial_acc_metric(self.model, adversarial_model, adv_params)
                # DATAA_acc_metric = get_DATAA_acc_metric(self.model, adversarial_model, adv_params)
                # metrics = ['accuracy', adv_acc_metric]
            # else:
            print('#### Clean Acc ####')
            metrics= ['accuracy']

        # compile the model
        self.model.compile(loss = loss,
                           optimizer = opt,
                           metrics = metrics)

        # training the model
        if dataset.name in ['cifar10', 'cifar100']:
            print('#### DATA AUGMENTATION in {}####'.format(str(dataset.name)))
            aug = keras.preprocessing.image.ImageDataGenerator(
                featurewise_center= False, samplewise_center= False,
                featurewise_std_normalization= False, samplewise_std_normalization= False,
                zca_whitening= False, rotation_range= 0,
                width_shift_range= 0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range= 0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip= True,  # randomly flip images
                vertical_flip= False  # randomly flip images
            )
            # zoom，(noise), (scale)，(contrast)
            def para_1(xtrain= dataset.X_train, ytrain= dataset.Y_train):
                aug.fit(xtrain)
                gen = aug.flow(xtrain, ytrain, batch_size= batch_size)
                return {"generator": gen,
                        "steps_per_epoch": len(gen)}  # xtrain.shape[0]/batch_size,
            model_fit = self.model.fit_generator
        elif dataset.name in ['mnist']:
            def para_1(xtrain= dataset.X_train, ytrain= dataset.Y_train):
                return {"x": xtrain, "y": ytrain,
                        "batch_size": batch_size}
            model_fit = self.model.fit

        model_fit(**para_1(dataset.X_train, dataset.Y_train),
                  epochs= nb_epochs,
                  verbose= 2,
                  callbacks= callbacks,
                  validation_data=(dataset.X_test, dataset.Y_test))

        acc_all = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        if attack is not None:
                loss, acc, adv_acc = acc_all
                print('#### Test Loss: %.4f####Test Accuracy:%.4f####Adv Accuracy:%.4f ####' % (loss, acc, adv_acc))
        else:
            # if adv_train == 'bat':
            #     loss, adv_acc, norm = acc_all
            #     print('#### Test Loss: %.4f#####Adv Accuracy:%.4f #### Norm: %.4f'
            #           % (loss, adv_acc, norm))
                # loss, acc, adv_acc, norm = acc_all
                # print('#### Test Loss: %.4f####Test Accuracy:%.4f####Adv Accuracy:%.4f #### Norm: %.4f'
                #       % (loss, acc, adv_acc, norm))
            # else:
            loss, acc = acc_all
            print('#### Test Loss: %.4f####Test Accuracy:%.4f ####' % (loss, acc))

        # strip model from pruning wrapper
        # if prun_para is not None:
        #     self.model = strip_pruning(self.model)
        #
        # # Save the original model
        # if adv_train is not None:
        #     if not os.path.exists('./'+ str(dataset.name) + '/' + str(adv_train) + '_' + str(eps)):
        #         os.makedirs('./' + str(dataset.name) + '/' + str(adv_train) + '_' + str(eps))
        #     keras.models.save_model(self.model,
        #                             './' + str(dataset.name) + '/' +str(adv_train) + '_' + str(eps) +'/'+ self.keras_file,
        #                             include_optimizer = False)
        #     print('save model in {}'.format
        #           ('./' + str(dataset.name) + '/' +str(adv_train) + '_' + str(eps) +'/'+ self.keras_file))
        # else:
        #     if not os.path.exists('./' + str(dataset.name) + '/NT'):
        #         os.makedirs('./' + str(dataset.name) + '/NT')
        #     keras.models.save_model(self.model, './' + str(dataset.name) + '/NT/' + self.keras_file,
        #                             include_optimizer = False)
        #     print('save model in {}'.format
        #           ('./' + str(dataset.name) + '/NT/' + self.keras_file))


if __name__ == '__main__':

    from data_load import DataSet
    from keras.optimizers import Adam, SGD

    from cleverhans.attacks import FastGradientMethod
    from cleverhans.utils_keras import cnn_model

    dataset = DataSet(name='mnist', label_smoothing=0.1)
    model = cnn_model(img_rows=28, img_cols=28,
                    channels=1, nb_filters=64,
                    nb_classes=10)

    # from Model import model

    # model = model('test', img_rows=dataset.img_rows, img_cols=dataset.img_cols,
    #               img_channels=dataset.img_channels, nb_classes=dataset.num_classes)
    sess = tf.Session()
    keras.backend.set_session(sess)

    wrap = KerasModelWrapper(model)
    fgsm = FastGradientMethod(wrap, sess=sess)
    fgsm_params = {'eps': 0.3,
                   'clip_min': 0.,
                   'clip_max': 1.}
    #
    adv_acc_metric = get_adversarial_acc_metric(model, fgsm, fgsm_params)
    adv_loss = get_adversarial_loss(model, fgsm, fgsm_params)

    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        # optimizer='adam',
        # loss=keras.losses.categorical_crossentropy,
        loss = adv_loss,
        metrics=['accuracy', adv_acc_metric]
    )

    model.fit(dataset.X_train, dataset.Y_train,
                       batch_size= 128,
                       epochs= 2,
                       verbose= 2,
                       validation_data=(dataset.X_test, dataset.Y_test))

    #loss, accuracy = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
    acc_all = model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
    #print('######Test Loss: %.4f####Test Accuracy:%.4f######' % (loss, accuracy))
    print(acc_all)