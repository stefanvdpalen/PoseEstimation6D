def train_net(self,train_generator,validation_generator,path_pesos):

        Autoencoder.compile(optimizer = tf.keras.optimizers.Adam(0.0002),
                            loss = tf.keras.losses.MeanSquaredError(),
                            metrics = tf.keras.losses.MeanSquaredError())

        callbacks = self.__callbacks_def(path_pesos)

        self.__full_model.fit(x = train_generator,
                              validation_data = validation_generator,
                              epochs=1000,
                              verbose=1,
                              callbacks=callbacks)

    def __callbacks_def(self,path_pesos):

        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50 ,verbose=1)

        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath = path_pesos,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True)

        callbacks = [tf.keras.callbacks.TensorBoard(log_dir='./logs'),
                    es,
                    model_checkpoint_callback]

        return callbacks


    def inference(self,img):
        return self.__inference_net.predict(img)

    def generation(self,encoding):
        return self.__generative_net.predict(encoding)
