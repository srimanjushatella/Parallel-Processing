
# coding: utf-8

# In[1]:


# imports needed
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import PIL.Image


# In[2]:


tf.__version__


# In[3]:


import vgg16


# In[1]:


vgg16.maybe_download()


# In[5]:


def load_image(filename, max_size=None):
    image = PIL.Image.open(filename)

    if max_size is not None:
        
        # ensuring max height and width, while keeping
        factor = max_size / np.max(image.size)
    
        # Scaling the image's height and width.
        size = np.array(image.size) * factor

        # The size is now floating-point because of scaling.
        # casting the size to be integers as per PIL.
        size = size.astype(int)

        # Resizing the image.
        image = image.resize(size, PIL.Image.LANCZOS)

    # Converting to numpy floating-point array.
    return np.float32(image)


# In[6]:


def save_image(image, filename):
    # Pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)
    
    # Converting them to bytes.
    image = image.astype(np.uint8)
    
    # Writing the image-file in jpeg-format.
    with open(filename, 'wb') as file:
        PIL.Image.fromarray(image).save(file, 'jpeg')


# In[7]:


def plot_image_big(image):
    # Pixel-values are between 0 and 255.
    image = np.clip(image, 0.0, 255.0)

    # Converting pixels to bytes.
    image = image.astype(np.uint8)

    # Converting to PIL-image and displaying it.
    display(PIL.Image.fromarray(image))


# In[8]:


def plot_images(content_image, style_image, mixed_image):
    # Creating the sub-plots.
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Adjusting the spacing.
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    # Smoothing pixels
    smooth = True
    
    # Type of interpolation being used.
    if smooth:
        interpolation = 'sinc'
    else:
        interpolation = 'nearest'

    # Plotting of the content-image.
    # Normalixation of the pixel images.
    ax = axes.flat[0]
    ax.imshow(content_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Content")

    # Plotting the mixed-image.
    ax = axes.flat[1]
    ax.imshow(mixed_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Mixed")

    # Plotting the style-image
    ax = axes.flat[2]
    ax.imshow(style_image / 255.0, interpolation=interpolation)
    ax.set_xlabel("Style")

    # Removing ticks from all the plots.
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    #Showing the plot.
    plt.show()


# In[9]:


def mean_squared_error(a, b):
    return tf.reduce_mean(tf.square(a - b))


# In[10]:


def create_content_loss(session, model, content_image, layer_ids):
    """
    Create the loss-function for the content-image.
    
    Parameters:
    session: An open TensorFlow session for running the graph.
    model: The model, here an instance of the VGG16-class.
    content_image: Numpy float array of the content-image.
    layer_ids: List of integers id's for the layers to be used in the model.
    """
    
    # Creating a feed-dict related to the content-image.
    feed_dict = model.create_feed_dict(image=content_image)

    # Getting the references to the tensors for the given layers.
    layers = model.get_layer_tensors(layer_ids)

    # Calculating the output values when content-image is feeded to the model.
    values = session.run(layers, feed_dict=feed_dict)

    # Set the model's graph as the default so we can add computational nodes to it. 
    with model.graph.as_default():
        
        # Initializing an empty list of loss functions.
        layer_losses = []
    
        # this is for the content-image.
        for value, layer in zip(values, layers):
            #It should be a constant value
            value_const = tf.constant(value)

            # The loss function of this layer is the MSE (Mean Squared Error) between the layer values
            loss = mean_squared_error(layer, value_const)

            # Now adding this loss function of this layer to the empty list of loss functions initialized above
            layer_losses.append(loss)

        # Calculating the total loss which is an average value.
        total_loss = tf.reduce_mean(layer_losses)
        
    return total_loss


# In[11]:


def gram_matrix(tensor):
    shape = tensor.get_shape()
    
    # The number of feature channels.
    num_channels = int(shape[3])

    # Reshaping the tensor so it is a 2-dimension matrix.
    matrix = tf.reshape(tensor, shape=[-1, num_channels])
    
    # Calculating the Gram-matrix. 
    # Gram matrix is the multiplication product of itself.
    gram = tf.matmul(tf.transpose(matrix), matrix)

    return gram                            


# In[12]:


def create_style_loss(session, model, style_image, layer_ids):
    """
    Create the loss-function for the style-image.
    
    Parameters:
    session: An open TensorFlow session for running the graph.
    model: The model, here an instance of the VGG16-class.
    style_image: Numpy float array with the style-image.
    layer_ids: List of integers id's for the layers to be used in the model.
    """

    # Creating a feed-dict with the style-image.
    feed_dict = model.create_feed_dict(image=style_image)

    # Getting the references to the tensors.
    layers = model.get_layer_tensors(layer_ids)

   
    with model.graph.as_default():
        
        # Calculating the Gram-matrices for each layer.
        gram_layers = [gram_matrix(layer) for layer in layers]

        # CAlculating when feeding the style image to the model.
        values = session.run(gram_layers, feed_dict=feed_dict)

        # Initializing an empty list of loss functions.
        layer_losses = []
    
        # For each Gram-matrix layer.
        for value, gram_layer in zip(values, gram_layers):
            #It should be a constant value.
            value_const = tf.constant(value)

            # The loss function value is the MSE between the two.
            loss = mean_squared_error(gram_layer, value_const)

            # Adding this loss function to the intialized loss functions list.
            layer_losses.append(loss)

        
        total_loss = tf.reduce_mean(layer_losses)
        
    return total_loss


# In[13]:


# This function sums over all the pixels by taking the difference between the original image. 
# And the image obtained by shifting one oixel in both x-axis and y-axis.
def create_denoise_loss(model):
    loss = tf.reduce_sum(tf.abs(model.input[:,1:,:,:] - model.input[:,:-1,:,:])) +            tf.reduce_sum(tf.abs(model.input[:,:,1:,:] - model.input[:,:,:-1,:]))

    return loss


# In[14]:


def style_transfer(content_image, style_image,
                   content_layer_ids, style_layer_ids,
                   weight_content=1.5, weight_style=10.0,
                   weight_denoise=0.3,
                   num_iterations=120, step_size=10.0):

    # Creating an instance of the pretrained VGG16-model. 
    model = vgg16.VGG16()

    # Creating a TensorFlow session.
    session = tf.InteractiveSession(graph=model.graph)

    # This is for printing the name of the content-layers.
    print("Content layers:")
    print(model.get_layer_names(content_layer_ids))
    print()

    # This is for printing the name of the style-layers
    print("Style layers:")
    print(model.get_layer_names(style_layer_ids))
    print()

    # Creating the loss function for the content layers.
    loss_content = create_content_loss(session=session,
                                       model=model,
                                       content_image=content_image,
                                       layer_ids=content_layer_ids)

    # Creating the loss function for the style layers.
    loss_style = create_style_loss(session=session,
                                   model=model,
                                   style_image=style_image,
                                   layer_ids=style_layer_ids)    

    # Create the denoising loss function of the mixed-image.
    loss_denoise = create_denoise_loss(model)

      #tensor flow variables to adjuxt the loss functions
    adj_content = tf.Variable(1e-10, name='adj_content')
    adj_style = tf.Variable(1e-10, name='adj_style')
    adj_denoise = tf.Variable(1e-10, name='adj_denoise')

    # Initializing of the values for loss functions.
    session.run([adj_content.initializer,
                 adj_style.initializer,
                 adj_denoise.initializer])

   
    update_adj_content = adj_content.assign(1.0 / (loss_content + 1e-10))
    update_adj_style = adj_style.assign(1.0 / (loss_style + 1e-10))
    update_adj_denoise = adj_denoise.assign(1.0 / (loss_denoise + 1e-10))

   
    loss_combined = weight_content * adj_content * loss_content +                     weight_style * adj_style * loss_style +                     weight_denoise * adj_denoise * loss_denoise

    # tensor flow function.
    gradient = tf.gradients(loss_combined, model.input)

    run_list = [gradient, update_adj_content, update_adj_style,                 update_adj_denoise]

    # The output image is started with a random noise.
    mixed_image = np.random.rand(*content_image.shape) + 128

    for i in range(num_iterations):
        # creating a feed dictionary.
        feed_dict = model.create_feed_dict(image=mixed_image)

        
        grad, adj_content_val, adj_style_val, adj_denoise_val         = session.run(run_list, feed_dict=feed_dict)

        grad = np.squeeze(grad)

        step_size_scaled = step_size / (np.std(grad) + 1e-8)

        # Updating the image.
        mixed_image -= grad * step_size_scaled

        # Checking the image has pixel values between 0 and 255.
        mixed_image = np.clip(mixed_image, 0.0, 255.0)

        print(". ", end="")

        # Displaying status.
        if (i % 10 == 0) or (i == num_iterations - 1):
            print()
            print("Iteration:", i)

            
            msg = "Weight Adj. for Content: {0:.2e}, Style: {1:.2e}, Denoise: {2:.2e}"
            print(msg.format(adj_content_val, adj_style_val, adj_denoise_val))

            # Plotting the images.
            plot_images(content_image=content_image,
                        style_image=style_image,
                        mixed_image=mixed_image)
            
    print()
    print("Final image:")
    plot_image_big(mixed_image)

    # Closing the TensorFlow session.
    session.close()
    
    return mixed_image


# In[15]:


# Example-1:
# Loading a content image of our choice.
content_filename = 'images/panda.jpg'
content_image = load_image(content_filename, max_size=None)


# In[16]:


#Loading a style image of our choice.
style_filename = 'images/fall.jpg'
style_image = load_image(style_filename, max_size=300)


# In[17]:


#defining the layer number.
content_layer_ids = [4]


# In[18]:


# The VGG16-model consists of 13 layers.
style_layer_ids = list(range(13))


# In[19]:


# Declaring the number of iterations and other parameters needed for the style transfer algorithm.
%%time
img = style_transfer(content_image=content_image,
                     style_image=style_image,
                     content_layer_ids=content_layer_ids,
                     style_layer_ids=style_layer_ids,
                     weight_content=1.5,
                     weight_style=10.0,
                     weight_denoise=0.3,
                     num_iterations=60,
                     step_size=10.0)


# In[20]:


# Example-2 :
# Load a content image of your choice.
# this is am image of my Professor (Dr. Pablo Rivas)
content_filename = 'images/prof.jpg'
content_image = load_image(content_filename, max_size=None)


# In[21]:


## Load a style image of your choice.
style_filename = 'images/tropical.jpg'
style_image = load_image(style_filename, max_size=300)


# In[22]:


content_layer_ids = [4]


# In[23]:


style_layer_ids = list(range(13))


# In[24]:


get_ipython().run_cell_magic('time', '', 'img = style_transfer(content_image=content_image,\n                     style_image=style_image,\n                     content_layer_ids=content_layer_ids,\n                     style_layer_ids=style_layer_ids,\n                     weight_content=1.5,\n                     weight_style=10.0,\n                     weight_denoise=0.3,\n                     num_iterations=1000,\n                     step_size=10.0)')


# In[25]:


# Example-3 :
# Load a content image of your choice.
# This is an image of myself.
content_filename = 'images/me1.jpg'
content_image = load_image(content_filename, max_size=None)


# In[26]:


## Load a style image of your choice.
style_filename = 'images/abstract.jpg'
style_image = load_image(style_filename, max_size=300)


# In[27]:


content_layer_ids = [4]


# In[28]:


style_layer_ids = list(range(13))


# In[29]:


get_ipython().run_cell_magic('time', '', 'img = style_transfer(content_image=content_image,\n                     style_image=style_image,\n                     content_layer_ids=content_layer_ids,\n                     style_layer_ids=style_layer_ids,\n                     weight_content=1.5,\n                     weight_style=10.0,\n                     weight_denoise=0.3,\n                     num_iterations=300,\n                     step_size=10.0)')

