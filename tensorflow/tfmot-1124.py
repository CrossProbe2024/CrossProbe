def build_model(args, include_preprocessing=False):
    IMG_SHAPE = (args.input_dim, args.input_dim, 3)
    # Transfer learning model with MobileNetV3
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights='imagenet',
        minimalistic=True,
        include_preprocessing=include_preprocessing
    )
    # Freeze the pre-trained model weights
    base_model.trainable = False
    cl1 = CustomLayer()(base_model.output)
    cl1 = tf.keras.layers.Dropout(0.2, name="dropout_cl1")(cl1)
    
    cl2 = CustomLayer()(base_model.output)
    cl2 = tf.keras.layers.Dropout(0.2, name="dropout_gd2")(gd2)
    
    cl3 = CustomLayer()(base_model.output)
    cl3 = tf.keras.layers.Dropout(0.2, name="dropout_cl1")(cl3)

    concat_cls = tf.keras.layers.Concatenate()([cl1, cl2, cl3])

    x = tf.keras.layers.Dense(512, activation='swish')(concat_cls) # No activation on final dense layer
    model = tf.keras.Model(base_model.input, x)
    
    return model

# This is initial training loop before QAT --> this returns few epoch trained model
model = build_model(args)
model = initial_training(args, model)

def apply_quantization_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.quantization.keras.quantize_annotate_layer(layer)
    return layer

annotated_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_quantization_to_dense,
)

quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)

