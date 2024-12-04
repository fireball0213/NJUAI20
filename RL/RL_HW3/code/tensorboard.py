import tensorflow as tf

#simple demo
# 定义一个计算图，实现两个向量的加法
# 定义两个输入，a为常量，b为随机值
a=tf.constant([10.0, 20.0, 40.0], name='a')
b=tf.Variable(tf.random.uniform([3]), name='b')   # 从均匀分布中输出随机值,[3]代表张量尺寸
output=tf.add_n([a,b], name='add')    #Add all input tensors element wise

# 如果你想继续使用 session 的方式，可以启用兼容模式
tf.compat.v1.disable_eager_execution()

# 如果你想继续使用 session 的方式，可以启用兼容模式
tf.compat.v1.disable_eager_execution()

with tf.compat.v1.Session() as sess:
    # 生成一个具有写权限的日志文件操作对象，将当前命名空间的计算图写进日志
    writer = tf.compat.v1.summary.FileWriter('D:/tf_dir/tensorboard_study', sess.graph)
    sess.run(tf.compat.v1.global_variables_initializer())
    f = sess.run(output)
    print(f)
    writer.close()