'''
Functionality for dealing with a neural network hessian. This includes methods
for computing the hessian vector product, as well as sub-sampling methods.
'''

'''
Methods for computing hessian vector products via forward and backward automatic
differentiation.

Only the backward-backward mode operator can compute multiple hvps without
needing to do another forward pass through the network. However, this makes it
more memory intensive.
'''
from tensorflow.python.eager.forwardprop import ForwardAccumulator

'''Single evaluation forward-backward mode'''
# def _forward_over_back_hvp(model, images, labels, vector):
#     with ForwardAccumulator(model.trainable_variables, vector) as acc:
#         with tf.GradientTape() as grad_tape:
#             logits = model(images, training=True)
#             loss = tf.compat.v1.losses.softmax_cross_entropy(
#               logits=logits, onehot_labels=labels)
#
#     grads = grad_tape.gradient(loss, model.trainable_variables)
#     return acc.jvp(grads)

'''Single evaluation backward-forward mode'''
# def _back_over_forward_hvp(model, images, labels, vector):
#     with tf.GradientTape() as grad_tape:
#         grad_tape.watch(model.trainable_variables)
#
#         with ForwardAccumulator(model.trainable_variables, vector) as acc:
#             logits = model(images, training=True)
#             loss = tf.compat.v1.losses.softmax_cross_entropy(
#               logits=logits, onehot_labels=labels)
#
#   return grad_tape.gradient(acc.jvp(loss), model.trainable_variables)

'''
Multiple evaluation backward-backward mode

Use:
    grad, hvp = back_over_back(model, inputs, labels)
    hvp(vector)
'''
def back_over_back(model, inputs, labels):
    with tf.GradientTape(persistent=True) as outer_tape:
        with tf.GradientTape() as inner_tape:
            outputs = model(inputs, training=True)
            loss = model.compiled_loss(inputs, labels)

        grads = inner_tape.gradient(loss, model.trainable_variables)

    def hvp(vector):
        return outer_tape.gradient(grads, model.trainable_variables,
                                    output_gradients=vector)

    return grads, hvp

'''
Methods for hessian sub-sampling.
'''
