def architectures_to_test():
    """Returns list of architectures to test"""
    return [
        ('Imagene',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': True,
             'max_pooling': True,
             'convolution': True,
             'filters': 32,
             'depth': 3,
             'kernel_height': 3,
             'kernel_width': 3,
             'num_dense_layers': 1
         }
         ),


        ('1 Conv Layers (1 2x1 kernel) + Relu -> Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': True,
             'max_pooling': False,
             'convolution': True,
             'filters': 1,
             'depth': 1,
             'kernel_height': 2,
             'kernel_width': 1,
             'num_dense_layers': 0
         }
         ),
    ]
