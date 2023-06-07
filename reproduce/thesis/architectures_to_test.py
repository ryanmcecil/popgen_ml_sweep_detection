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

<<<<<<< HEAD
=======
        ('3 Conv Layers (32 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid',
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
             'num_dense_layers': 0
         }
         ),

        ('2 Conv Layers (32 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': True,
             'max_pooling': True,
             'convolution': True,
             'filters': 32,
             'depth': 2,
             'kernel_height': 3,
             'kernel_width': 3,
             'num_dense_layers': 0
         }
         ),

        ('1 Conv Layers (32 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': True,
             'max_pooling': True,
             'convolution': True,
             'filters': 32,
             'depth': 1,
             'kernel_height': 3,
             'kernel_width': 3,
             'num_dense_layers': 0
         }
         ),

        ('1 Conv Layers (16 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': True,
             'max_pooling': True,
             'convolution': True,
             'filters': 16,
             'depth': 1,
             'kernel_height': 3,
             'kernel_width': 3,
             'num_dense_layers': 0
         }
         ),

        ('1 Conv Layers (4 3x3 kernels) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': True,
             'max_pooling': True,
             'convolution': True,
             'filters': 4,
             'depth': 1,
             'kernel_height': 3,
             'kernel_width': 3,
             'num_dense_layers': 0
         }
         ),

        ('1 Conv Layers (1 3x3 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': True,
             'max_pooling': True,
             'convolution': True,
             'filters': 1,
             'depth': 1,
             'kernel_height': 3,
             'kernel_width': 3,
             'num_dense_layers': 0
         }
         ),

        ('1 Conv Layers (1 2x2 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': True,
             'max_pooling': True,
             'convolution': True,
             'filters': 1,
             'depth': 1,
             'kernel_height': 2,
             'kernel_width': 2,
             'num_dense_layers': 0
         }
         ),

        ('1 Conv Layers (1 1x2 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': True,
             'max_pooling': True,
             'convolution': True,
             'filters': 1,
             'depth': 1,
             'kernel_height': 1,
             'kernel_width': 2,
             'num_dense_layers': 0
         }
         ),

        ('1 Conv Layers (1 2x1 kernel) + Max Pooling + Relu -> Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': True,
             'max_pooling': True,
             'convolution': True,
             'filters': 1,
             'depth': 1,
             'kernel_height': 2,
             'kernel_width': 1,
             'num_dense_layers': 0
         }
         ),
>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83

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
<<<<<<< HEAD
=======

        ('1 Conv Layers (1 2x1 kernel) + Max Pooling-> Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': False,
             'max_pooling': True,
             'convolution': True,
             'filters': 1,
             'depth': 1,
             'kernel_height': 2,
             'kernel_width': 1,
             'num_dense_layers': 0
         }
         ),

        ('1 Conv Layers (1 2x1 kernel) -> Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': False,
             'max_pooling': False,
             'convolution': True,
             'filters': 1,
             'depth': 1,
             'kernel_height': 2,
             'kernel_width': 1,
             'num_dense_layers': 0
         }
         ),

        ('Dense (1 unit) + Sigmoid',
         {
             'type': 'ml',
             'name': 'imagene',
             'image_height': 128,
             'image_width': 128,
             'relu': False,
             'max_pooling': False,
             'convolution': False,
             'filters': 1,
             'depth': 1,
             'kernel_height': 2,
             'kernel_width': 1,
             'num_dense_layers': 0
         }
         ),


>>>>>>> 3ab9be5a0e89e17043c9d1df756f03b0d458ce83
    ]
