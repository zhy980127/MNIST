import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def train_mnist():
    """
    训练mnist数字集进行数字识别
    :return: None
    """
    #加载mnist数字集
    mnist = input_data.read_data_sets("D:/my_test/MNIST/MNIST_data/", one_hot=True)
    with tf.variable_scope("data"):
        #1.创建数据
        x_data = tf.placeholder(tf.float32,[None,784],"x_data")
        y_data = tf.placeholder(tf.int32,[None,10],"y_data")
    with tf.variable_scope("model"):
        #权重，偏置
        weight = tf.Variable(tf.random_normal([784,10],mean=0.0,stddev=1.0),name="weight")
        bias = tf.Variable(0.0,name="bias")
        #2.创建模板
        y_true = tf.matmul(x_data,weight) + bias
    with tf.variable_scope("soft_cross"):
        #3.求平均交叉熵损失(求出所有样本的损失 求均值)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data,logits=y_true))
    with tf.variable_scope("train_op"):
        #4.梯度下降损失
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    with tf.variable_scope("acc"):
        #计算准确率
        equal_list = tf.equal(tf.arg_max(y_data,1),tf.argmax(y_true,1))
        accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))
    # 收集tensor,收集对于损失函数和准确率等单值变量
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("accuracy",accuracy)
    # 收集高纬度参数
    tf.summary.histogram("weight", weight)
    tf.summary.histogram("bias", bias)
    # 合并写入事件文件
    merged = tf.summary.merge_all()
    #初始化op
    init_op = tf.global_variables_initializer()
    # 保存模型
    saver = tf.train.Saver(max_to_keep=5)

    #运行程序
    with tf.Session() as sess:
        sess.run(init_op)
        summary_filr = tf.summary.FileWriter("./test",graph=sess.graph)
        if os.path.exists("D:\my_test\MNIST\ckpt\checkpoint"):
            # 加载模型
            saver.restore(sess, "D:\my_test\MNIST\ckpt\model")
            for i in range(100):
                x_test, y_test = mnist.train.next_batch(1)
                print("第%d张图片，目标是%d的，预测结果是%d" % (i, tf.argmax(y_test, 1).eval(),
                                                 tf.argmax(sess.run(y_true, feed_dict={x_data: x_test, y_data: y_test}),
                                                           1).eval()
                                                 ))
        else:
            for i in range(5000):
                mnist_x,mnist_y = mnist.train.next_batch(50)
                feeds = {x_data: mnist_x, y_data: mnist_y}
                summary = sess.run(merged,feed_dict=feeds)
                summary_filr.add_summary(summary,i)
                sess.run(train_op,feed_dict=feeds)
                train_acc = sess.run(accuracy,feed_dict=feeds)
                print("训练第%d步,准确率%f"%(i,train_acc))
            saver.save(sess,"D:\my_test\MNIST\ckpt\model")

    return None

if __name__ == "__main__":
    train_mnist()
