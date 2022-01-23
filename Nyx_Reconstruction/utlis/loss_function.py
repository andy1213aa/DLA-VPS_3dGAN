import tensorflow as tf


def generator_loss(gfake_logit, type):
  
    # if type == 'WGAN' or 'hinge':
    g_loss = -tf.reduce_mean(gfake_logit)

    # #type == 'origin':
    # g_loss = -tf.reduce_mean(tf.math.log(gfake_logit))
 
    
    return  g_loss

def discriminator_loss(real_logit, fake_logit, type):
    # if type == 'WGAN':
    real_loss = -tf.reduce_mean(real_logit)
    fake_loss = tf.reduce_mean(fake_logit)
    
    # elif type == 'hinge':
    # real_loss =  tf.reduce_mean(tf.nn.relu(1.0 - real_logit))
    # fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake_logit))
 
    # # type == 'origin':
    # real_loss = -tf.reduce_mean(tf.math.log(real_logit))
    # fake_loss = -tf.reduce_mean(tf.math.log(1-fake_logit))

    return real_loss, fake_loss


    