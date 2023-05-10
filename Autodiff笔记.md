# Autodiff笔记

 官网：[Installation - autodiff](https://autodiff.github.io/installation/)

还没来得及整理，过了这段比较忙的时间再整理

## **Forward mode**

```c++
// 包含的头文件，还没写


using namespace autodiff
    
dual f(const dual& x, const dual& y, const dual& z)
{
    return (x + y + z) * exp(x * y * z);
}

dual x = 1.0;
dual y = 2.0;
dual z = 3.0;
dual u = f(x, y, z);

double dudx = derivative(f, wrt(x), at(x, y, z));
double dudy = derivative(f, wrt(y), at(x, y, z));
double dudz = derivative(f, wrt(z), at(x, y, z));
```

