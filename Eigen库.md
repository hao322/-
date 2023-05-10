# Eigen库

​		由于摄影测量以及专业的需要，eigen库是十分重要的，因此写下这篇学习笔记以便学习和巩固

本人的水平十分的有限，本文中也主要是作业中常用以及会用到的一些操作。

​		如果你想更好地了解Eigen库，这里附上官方学习文档：

官方文档： [Eigen: Main Page](http://eigen.tuxfamily.org/dox/)（强推！！！最官方最详细最好的）

官方链接：http://eigen.tuxfamily.org/index.php?title=Main_Page

------

## 矩阵类Matrix

​		Matrix 类总共有六个模板参数首先只介绍前三个参数，剩下的三个参数有其默认值。三个强制型的参数如下：

```c++
Matrix<typename Scalar,int Row,int Col>
```

​		Scalar是scalar类型，如想要构造一个单精度的浮点类型矩阵，可以选择folat。所有支持的Scalar类型参见

​		Row和Col分别表示行数和列数，要求这两个参数在编译时已知。

​		在Eigen中提供了宏定义来便捷的访问一些常用的类型，如：

```c++
typedef Matrix<float,4,4> Matrix4f;
typedef Matrix<float,3,1> Vector3f;
typedef Matrix<int,1,2> RowVector2i;
```

​		当然，Eigen并不局限于那些矩阵维数在编译时已知的情形。Row和Col可以取一个特殊的值Dynamic，这表示矩阵的维数在编译时是未知的，必须作为一个运行时变量来处理。在Eigen术语中，Dynamic称之为动态大小（dynamic size），而运行时已知的大小称之为固定大小（fixed）。

```c++
//创建一个双精度的动态矩阵
typedef Matrix<double, Dynamic, Dynamic> MatrixXd;
//创建一个整型列向量
typedef Matrix<int, Dynamic, 1> VectorXi;
```

​		Matrix 另外的三个参数（）不再介绍。

### 矩阵的初始化与访问

​		Eigen提供了默认构造函数，它不会提供动态内存的分配，也不会初始化任何矩阵的值。Eigen类型可以这样使用：

```c++
// 一个3*3的矩阵，矩阵的元素都没有被初始化
Matrix3f a;
// 一个动态矩阵，它的大小是0*0，也就是还没有为该矩阵分配内存
MatrixXf b;
```

​		构造函数提供指定矩阵的大小的重载。对矩阵来说，第一个参数是矩阵的行数。对向量来说只需指定向量的大小。它会分配矩阵或向量所需的内存的大小，但不会初始化它们的值。

```c++
// 一个10*15的动态矩阵，内存进行了分配，但没有初始化
MatrixXf a(10,15);
// 一个大小为30的动态数组，但没有初始化
VectorXf b(30);
```

​		Eigen库重载圆括号()访问矩阵或向量的元素，序号从0开始。Eigen不支持使用[]访问矩阵的元素（向量除外）。

```c++
MatrixXd m(2,2);
m(0,0) = 3;
VectorXd v(2);
v(0) = 4;
```

​		逗号表达式初始化

```c++
Matrix3f m; // 一个3*3的矩阵
m << 1,2,3,
	 4,5,6,
	 7,8,9;
std::cout << m;
```

​		可以使用rows()、cols()、size()访问矩阵当前大小（例如：m.rows()），使用resize()重置矩阵的大小。（更深入的这里不再赘述）

### 矩阵和向量代数

​		**加减法：**重载C++中”+“、”-“、”+=“、”-=“操作符，要求左右操作数的维度相同。不允许一个向量加上或减去一个数。只要维度相同，向量和矩阵是可以相加的，这也印证了向量是特殊的矩阵。

​		 **数乘与数除：**重载C++中”*“、”/“、” *= “、”/=“操作符，支持矩阵和向量乘以或除以一个数。

​		**转置与共轭：**转置、共轭、共轭转置分别通过transpose()、conjugate()、adjoint()实现。调用格式a.transpose()，a.conjugate()，a.adjoint()。这里可能会出现一些问题，考虑到经常会用到转置，还是再说明一下。对于实数而言，共轭没有任何影响，共轭转置等价于转置。使用a=a.transpose()可能会出现错误，这是因为Eigen在进行转置或者共轭操作时，会同时写左操作数，从而得到意想不到的结果。要实现这种功能可以使用 a.transposeInPlace()。类似的，也支持adjointInPlace()。

​		**矩阵-矩阵与矩阵-向量乘法：**由于在Eigen中向量只是特殊的矩阵，因此只需要重载”*“、” *= “即可实现矩阵和向量的乘法。如果你担心 m=m * m 会导致混淆，现在可以消除这个疑虑，因为Eigen以一种特殊的方式处理矩阵乘法，编译 m = m * m 时，作为

```c++
tmp = m*m;
m = tmp;
```

​		**点积和叉乘：**点积又可以称为内积，Eigen分别使用 dot()和 cross()来实现内积和向量积。叉乘只适用于三维向量。

```c++
Vector3d v(1,2,3);
Vector3d w(0,1,2);
v.dot(w);
v.cross(w);
```

​		**基础的代数计算：**mat.sum()计算所有矩阵元素的和，mat.pro()计算所有元素的连乘积，mat.mean()计算所有元素的平均值，mat.minCoeff()计算矩阵元素的最小值，mat.maxCoeff()计算矩阵元素的最大值，mat.trace()计算矩阵的迹。计算最大值和最小值的函数支持返回最大值和最小值的位置：



photo第二次作业发现胡老师用了如下的矩阵（向量）：

```c++
// 相平面坐标
std::vector<Vector2d> point2ds = {Vector2d(-86.15, -68.99), Vector2d(-53.40, 82.21), Vector2d(-14.78, -76.63), Vector2d(10.46, 64.43)};
// 控制点坐标
std::vector<Vector3d> point3ds = {Vector3d(36589.41, 25273.32, 2195.17), Vector3d(37631.08, 31324.51, 728.69), Vector3d(39100.97, 24934.98, 2386.50), Vector3d(40426.54, 30319.81, 757.31)};
```

有些不懂，导致他的作业没法继续做，于是自己就尝试着找文档，找到的文档都令我没搞懂，或者说没有找到准确的，于是自己试着做了一些测试，最终发现，这样是可以的

```c++
vector<Vector2d> L(4);
vector<Matrix<double, 2, 2> > A(4); //重点，就是为了尝试这个东西

    A[0](0) = 12;
    cout << point2ds[1] << endl;
    cout << A[0];
    return 0;
```

有人是这么想的（知乎：[C++：vector小指南（附带一些新手错误） - 知乎(zhihu.com)](https://zhuanlan.zhihu.com/p/336492399)）

```c++
二维（指定行数，固定列长）
int N = 5, M = 6;
vector<vector<int>>obj(N, vector<int>(M)); //定义二维动态数组5行6列
```

```c++
二维（指定行数，不固定列长）
int N = 5, M = 6;
vector<vector<int> > obj(N); //定义二维动态数组大小5行 
for (int i = 0; i < obj.size(); i++)//动态二维数组为5行(i+3)列，值全为0 
{
	obj[i].resize(i+3);
}
```







一些。。。

```c++
/*
    matrix_33 = Matrix3d::Random();                           //生成一个3*3的随机矩阵
    cout << "random matrix: n"<< matrix_33 << endl;
    cout << "transpose : n"<< matrix_33.transpose() << endl;  //转置
    cout << "sum :" << matrix_33.sum() << endl;                //求和
    cout << "trace : "<< matrix_33.trace() << endl;            //求迹
    cout << "time 10: n"<< 10 * matrix_33 << endl;            //数乘
    cout << "inverse : n"<< matrix_33.inverse() << endl;      //求逆
    cout << "det : n"<< matrix_33.determinant() << endl;      //求行列式
*/

/*
    //实现矩阵拼接
    Eigen::Matrix3d a;
    a<<1,2,3,
        4,5,6,
        7,8,9;
    Eigen::Matrix3d b;
    b<<4,5,6,
        7,8,9,
        10,11,12;
    Eigen::Matrix<double,6,3> c;
    c<<a,b;
*/
```

------

### 进阶1（平移，旋转等等，暂时还未涉及），先插个眼



### 进阶2（几何模块命令及实例）

使用Eigen库的几何模块时，需要声明头文件<Eigen/Geometry>，此模块支持进行四元数、欧拉角和旋转矩阵的运算。各种常见形式的表达方式如下所示：

```c++
Eigen::Matrix3d      //旋转矩阵（3*3）
Eigen::AngleAxisd    //旋转向量（3*1）
Eigen::Vector3d      //欧拉角（3*1）
Eigen::Quaterniond   //四元数（4*1）
Eigen::Isometry3d    //欧式变换矩阵（4*4）
Eigen::Affine3d      //放射变换矩阵（4*4）
Eigen::Projective3d  //射影变换矩阵（4*4）
```

**上述数据类型均为双精度(double)类型，若要改为单精度(float)类型，把最后的d改为f即可。**

四元数（链接）：



下面给出程序实例，看完实例应该能基本了解了几何变换的形式，若对相关数学知识不了解，可以相应去学习一下：

```c++
//本程序将演示Geometry几何模块的使用
#include <iostream>
#include <cmath>
using namespace std;
 
#include <Eigen/Core>
#include <Eigen/Geometry>
//Geometry模块提供了各种旋转和平移的表示
using namespace Eigen;
 
int main(int argc, char ** argv)
{
    //3d旋转矩阵可以直接使用Matrix3d或者Matrix3f
    Matrix3d rotation_matrix = Matrix3d :: Identity();
 
    //旋转向量使用AngleAxis，运算可以当做矩阵
    AngleAxisd rotation_vector(M_PI / 4, Vector3d(0,0,1));     //眼Z轴旋转45°
    cout.precision(3);                                         //输出精度为小数点后两位
    cout << "rotation matrix = n" << rotation_vector.matrix() << endl;
    //用matrix转换成矩阵可以直接赋值
    rotation_matrix = rotation_vector.toRotationMatrix();
 
    //使用Amgleanxis可以进行坐标变换
    Vector3d v(1, 0, 0);
    Vector3d v_rotated = rotation_vector * v;
    cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;
 
    //使用旋转矩阵
    v_rotated = rotation_matrix * v;
    cout << "(1,0,0) after rotation (by matrix) = " << v_rotated.transpose() << endl;
 
    //欧拉角：可以将矩阵直接转换成欧拉角
    Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0);       //按照ZYX顺序
    cout << "yaw pitch row = "<< euler_angles.transpose() << endl;
 
    //欧式变换矩阵使用Eigen::Isometry
    Isometry3d T = Isometry3d::Identity();      //实质为4*4的矩阵
    T.rotate(rotation_vector);                  //按照rotation_vector进行转化
    T.pretranslate(Vector3d(1, 3, 4));          //平移向量设为（1， 3， 4）
    cout << "Transform matrix = n" << T.matrix() <<endl;
 
    //变换矩阵进行坐标变换
    Vector3d v_transformed = T *v;
    cout << "v transormed =" << v_transformed.transpose() << endl;
 
    //四元数
    //直接把AngleAxis赋值给四元数，反之亦然
    Quaterniond q = Quaterniond(rotation_vector);
    cout << "quaternion from rotation vector = " << q.coeffs().transpose() << endl;
    q = Quaterniond(rotation_matrix);
    cout << "quaternion from rotation matrix = "<< q.coeffs().transpose() << endl;
 
    //使用四元数旋转一个向量，使用重载的乘法即可
    v_rotated = q * v;
    cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
    cout << "should be equal to " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl;
 
    return 0;
}
```











## 向量Vector(可变长动态数组)

### 简单介绍

1. 向量 vector 是一种对象实体, 能够容纳许多其他类型相同的元素, 因此又被称为容器。 与string相同, vector 同属于STL(Standard Template Library, 标准模板库)中的一种自定义的数据类型, 可以广义上认为是数组的增强版
2. 在使用它时, 需要包含头文件 vector, `#include<vector>`
3. vector 容器与数组相比其优点在于它能够根据需要随时自动调整自身的大小以便容下所要放入的元素

### 常用操作

```c++
//声明与初始化

vector<int> a ; //声明一个int型向量a

vector<int> a(10) ; //声明一个初始大小为10的向量

vector<int> a(10, 1) ; //声明一个初始大小为10且初始值都为1的向量

vector<int> b(a) ; //声明并用向量a初始化向量b

vector<int> b(a.begin(), a.begin()+3) ; //将a向量中从第0个到第2个(共3个)作为向量b的初始值

//输入输出可以使用cin,cout; 使用中括号[]
```

在元素的输出上, 还可以使用遍历器(又称迭代器)进行输出控制。在 `vector<int> b(a.begin(), a.begin()+3) ;` 这种声明形式中, (a.begin()、a.begin()+3) 表示向量起始元素位置到起始元素+3之间的元素位置

在上例中讲元素全部输出部分的代码就可以改写为:

```c++
//全部输出

vector<int>::iterator t ;

for(t=a.begin(); t!=a.end(); t++)

cout<<*t<<" " ;

// *t 为指针的间接访问形式, 意思是访问t所指向的元素值。
```

```c++
基本操作

1>. a.size() //获取向量中的元素个数

2>. a.empty() //判断向量是否为空

3>. a.clear() //清空向量中的元素

4>. 复制

a = b ; //将b向量复制到a向量中

5>. 比较

保持 ==、!=、>、>=、<、<= 的惯有含义 ;

如: a == b ; //a向量与b向量比较, 相等则返回1

6>. 插入 - insert

①、 a.insert(a.begin(), 1000); //将1000插入到向量a的起始位置前

②、 a.insert(a.begin(), 3, 1000) ; //将1000分别插入到向量元素位置的0-2处(共3个元素)

③、 vector<int> a(5, 1) ;

vector<int> b(10) ;

b.insert(b.begin(), a.begin(), a.end()) ; //将a.begin(), a.end()之间的全部元素插入到b.begin()前

7>. 删除 - erase

①、 b.erase(b.begin()) ; //将起始位置的元素删除

②、 b.erase(b.begin(), b.begin()+3) ; //将(b.begin(), b.begin()+3)之间的元素删除

8>. 交换 - swap

b.swap(a) ; //a向量与b向量进行交换
```

2023.05.03-------东西实在是太多了，也不可能去全学

------







2023.04.25

今天在网上随意冲浪时（当然是为了完成作业了！！！），看到了一些关于vector的用法介绍，想着对我也有些用处，便抄录下来。这教程挺拉的，以后再看吧。

原文链接：[C++ vector使用方法_w3cschool](https://www.w3cschool.cn/cpp/cpp-i6da2pq0.html)

------

在 C++ 中，vector 是一个十分有用的容器。它能够像容器一样存放各种类型的对象，简单地说，vector是一个能够存放任意类型的动态数组，能够增加和压缩数据。

C++ 中数组很坑，有没有类似 Python 中 list 的数据类型呢？类似的就是 vector！vector 是同一种类型的对象的集合，每个对象都有一个对应的整数索引值。和 string 对象一样，标准库将负责管理与存储元素相关的内存。我们把 vector 称为容器，是因为它可以包含其他对象。一个容器中的所有对象都必须是同一种类型的。

一、什么是vector？

向量（vector）是一个封装了动态大小数组的顺序容器（Sequence Container）。跟任意其它类型容器一样，它能够存放各种类型的对象。可以简单的认为，向量是一个能够存放任意类型的动态数组。

二、容器特性

1.顺序序列
顺序容器中的元素按照严格的线性顺序排序。可以通过元素在序列中的位置访问对应的元素。

2.动态数组
支持对序列中的任意元素进行快速直接访问，甚至可以通过指针算述进行该操作。操供了在序列末尾相对快速地添加/删除元素的操作。

3.能够感知内存分配器的（Allocator-aware）
容器使用一个内存分配器对象来动态地处理它的存储需求。

三、基本函数实现

1.构造函数

```c++
vector():创建一个空vector
vector(int nSize):创建一个vector，元素个数为nSize
vector(int nSize,const t& t):创建一个vector，元素个数为nSize，且值均为t
vector(const vector&):复制构造函数
vector(begin,end):复制[begin,,end）区间内另一个数组的元素到vector。符号是开闭区间
```

2.增加函数

```c++
void push_back(const T& x):向量尾部增加一个元素X
iterator insert(iterator it,const T& x):向量中迭代器指向元素前增加一个元素X
iterator insert(iterator it,int n,const T& x):向量中迭代器指向元素前增加n个相同的元素X
iterator insert(iterator it,const_iterator first,const_iterator lat):向量中迭代器指向元素前插入另一个相同类型向量的[first,last）间的数据
```

3.删除函数

```
iterator erase(iterator it):删除向量中迭代器指向元素
iterator erase(iterator first,iterator last):删除向量中[first,last)中元素
void pop_back():删除向量中最后一个元素
void clear():清空向量中所有元素
```

4.遍历函数

```c++
reference at(int pos):返回pos位置元素的引用
reference front():返回首元素的引用
reference back():返回尾元素的引用
iterator begin():返回向量头指针，指向第一个元素
iterator end():返回向量尾指针，指向向量最后一个元素的下一个位置
reverse_iterator rbegin():反向迭代器，指向最后一个元素
reverse_iterator rend():反向迭代器，指向第一个元素之前的位置
```

5.判断函数

```c++
bool empty() const:判断向量是否为空，若为空，则向量中无元素
```

6.大小函数

```c++
int size() const:返回向量中元素的个数
int capacity() const:返回当前向量所能容纳的最大元素值
int max_size() const:返回最大可允许的 vector 元素数量值
```

7.其他函数

```c++
void swap(vector&):交换两个同类型向量的数据
void assign(int n,const T& x):设置向量中前n个元素的值为x
void assign(const_iterator first,const_iterator last):向量中[first,last)中元素设置成当前向量元素
```



简单介绍：

```c++
vector<类型>标识符
vector<类型>标识符(最大容量)
vector<类型>标识符(最大容量,初始所有值)
Int i[5]={1,2,3,4,5}vector<类型>vi(I,i+2);//得到i索引值为3以后的值
vector< vector< int> >v; 二维向量//这里最外的<>要有空格。否则在比较旧的编译器下无法通过
```

**vector使用实例**















## 杂七杂八：

点——标量（scalar）

线——向量（[vector](https://so.csdn.net/so/search?q=vector&spm=1001.2101.3001.7020)）

面——矩阵（[matrix](https://so.csdn.net/so/search?q=matrix&spm=1001.2101.3001.7020)）

体——张量（[tensor](https://so.csdn.net/so/search?q=tensor&spm=1001.2101.3001.7020)）
