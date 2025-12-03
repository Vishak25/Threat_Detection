import tensorflow as tf

class MILScoringHead(tf.keras.Model):
    def __init__(self, units=[256, 128], name="mil_head"):
        super().__init__(name=name)
        self.noise = tf.keras.layers.GaussianNoise(0.01) # Add noise to features
        self.denses = []
        self.dropouts = []
        for idx, unit in enumerate(units):
            self.denses.append(tf.keras.layers.Dense(unit, activation="relu", name=f"dense_{idx}"))
            self.dropouts.append(tf.keras.layers.Dropout(0.4, name=f"dropout_{idx}")) # Increased dropout
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="score")

    def call(self, inputs, training=False):
        x = self.noise(inputs, training=training)
        for dense, drop in zip(self.denses, self.dropouts):
            x = dense(x)
            x = drop(x, training=training)
        return self.output_layer(x)

def build_mil_model(input_shape=(224, 224, 3)):
    """
    Builds a MIL model with ResNet50V2 backbone and MILScoringHead.
    The model takes a single image (or batch of images) and outputs a score.
    For MIL, we typically process a bag of instances (clips) and aggregate scores.
    This model represents the instance-level scorer.
    """
    # Backbone
    backbone = tf.keras.applications.ResNet50V2(
        include_top=False, 
        weights='imagenet', 
        pooling='avg',
        input_shape=input_shape
    )
    # Freeze backbone
    backbone.trainable = False
    
    # MIL Head
    mil_head = MILScoringHead()
    
    # Input
    inputs = tf.keras.Input(shape=input_shape)
    features = backbone(inputs, training=False)
    scores = mil_head(features)
    
    model = tf.keras.Model(inputs=inputs, outputs=scores, name="ResNet50_MIL")
    return model

if __name__ == "__main__":
    model = build_mil_model()
    model.summary()
