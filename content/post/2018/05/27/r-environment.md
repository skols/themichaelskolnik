---
title: "My Anaconda R environment"
date: 2018-05-31
slug: "my-r-environment"
tags: ["r", "anaconda"]
draft: true
---

## Using Jupyter Notebook with R by creating a new environment with Anaconda

In this post I want to explain the R environment that I've created. I use Anaconda (available [here](https://www.anaconda.com/download/)) for my Python work, though these steps should be the same with MiniConda.

### Creating a new environment

You can install the r-essentials package into your current Anaconda by running the following:

```python
    conda install -c r r-essentials
```

I prefer to create a new environment though and to do that in Anaconda, you run the following:
```python
    conda create -n my-r-env -c r r-essentials
```

One thing to note. If you don't have conda added to your PATH, you have to point to it directly, i.e.
```python
    C:\Users\username\Anaconda3\Scripts\conda create -n my-r-env -c r r-essentials
```

You have to explicitly say by going to the folder where Anaconda is installed.
