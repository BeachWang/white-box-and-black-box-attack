# README
## 白盒攻击
<ul>
  <li>模型：  
  模型共有5个隐藏层。  
  第一个隐藏层：5x5的卷积层，深度为32，步长为1，padding使得图像尺寸不变；  
  第二个隐藏层：2x2的max pooling层，步长为2，图像尺寸变为原来1/4；  
  第三个隐藏层：5x5的卷基层，深度为64，步长为1，图像尺寸保持不变；  
  第四个隐藏层：2x2的max pooling层，步长为2，图像尺寸变为原来1/4；  
  第五个隐藏层：1024个神经元的全连接层，采用0.4的dropout。  
  模型代码：model.py
  </li>
  <li>
  测试集上的正确率：0.91480
  </li>
  <li>
  方法：通过模型对要冒充的类型求loss，并对x求导，将梯度乘一个常数加到x上，循环1000轮。具体参考代码w_attack.py。
  </li>
  <li>
  攻击成功率：0.646
  </li>
  <li>
  成功攻击样例展示，第一行为原图，第二行为攻击之后的图像。  
    ![cmd-markdown-logo](https://img-blog.csdnimg.cn/20191102143738960.png)  
    ![cmd-markdown-logo](https://img-blog.csdnimg.cn/2019110214380492.png)
  </li>
</ul>
