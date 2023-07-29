\[教程\]从零开始快速编写Makefile
======================

- [\[教程\]从零开始快速编写Makefile](#教程从零开始快速编写makefile)
  - [依赖关系](#依赖关系)
  - [变量与函数](#变量与函数)
  - [伪目标](#伪目标)
  - [四种等号](#四种等号)
  - [环境变量](#环境变量)
  - [变量的嵌套使用](#变量的嵌套使用)
  - [条件判断](#条件判断)
  - [关于@与-](#关于与-)
  - [关于依赖中的头文件](#关于依赖中的头文件)
  - [make命令参数](#make命令参数)
  - [简单Makefile模板](#简单makefile模板)


本文将讲述如何从零开始快速为一个小型项目编写Makefile，适合刚开始接触make命令、想弄懂Makefile机制的新手。对于已经弄清楚通过Makefile编译项目机制的小伙伴，文末附上针对C语言项目的Makefile模板，通过套用模板，可以快速为小型项目编写一套编译脚本。

依赖关系
----

通过make命令进行代码工程编译，其实际过程就是按照开发者在Makefile文件中所描述的模块与模块、模块与源代码文件之间的依赖关系，将源代码文件编译成obj文件，再将obj文件链接成库文件或可执行程序文件的过程。

如果程序只有一个源代码文件main.c，那么下面一条命令就可以完成程序test的编译，其依赖关系非常简单，就是可执行程序test依赖于源文件main.c。

    gcc main.c -o test

但软件项目中，模块之间的依赖关系一般比较复杂，想要通过一条命令gcc命令完成项目的编译非常困难，并且不利于扩展和维护。

下面给出一个简单工程示例，如何用Makefile规则为其快速编写一套编译脚本。

![](https://pic4.zhimg.com/v2-d7ed0199af5277cf04982e8502c65dfb_r.jpg)

    # test依赖于libtest.a main.o
    test: libtest.a main.o
    	gcc main.o -ltest -L. -o test
    # main.o依赖于main.c
    main.o:main.c
    	gcc main.c -c -o main.o
    # libtest.a依赖于test1.o test2.o
    libtest.a: test1.o test2.o
    	ar -r libtest.a test1.o test2.o
    # test1.o依赖于test1.c
    test1.o:test1.c
    	gcc test1.c -c -o test1.o
    # test2.o依赖于test2.c
    test2.o:test2.c
    	gcc test2.c -c -o test2.o

![](https://pic1.zhimg.com/v2-a6f2da6a4fd1d66c6d2c7d0a6818c0dc_r.jpg)

将所有源文件和Makefile放在一个目录下，执行make命令，可以看到直接生成了可执行程序test。这里没有使用Makefile里面任何技巧，仅仅使用Makefile的规则描述了各个文件之间的依赖关系。

    target... : prerequisites ...
              command
              ......

target

可以是一个object file（目标文件），也可以是一个执行文件，还可以是一个标签（label）。对于标签这种特性，后面会介绍。

prerequisites

生成该target所依赖的文件，可以有多个依赖

command

该target要执行的命令（任意的shell命令）

当要生成目标test时，make工具会从test开始依次寻找依赖关系，由源文件逐步生成目标test。test依赖于libtest.a和main.o，或者说libtest.a和main.o是生成test的前置条件（ prerequisites），如果libtest.a和main.o存在，那么将直接使用命令"gcc main.o -ltest -L. -o test"生成test；如果libtest.a和main.o不存在，再向下遍历，寻找libtest.a和main.o的前置条件，去生成libtest.a和main.o，就这样一直寻找到可以满足的前置条件，逐步生成目标test。如果找到最底层，依然无法满足生成条件，那么就会报错“make：\*\*\* 没有明确目标并且找不到 makefile。停止”。

在检查依赖关系时，同时会检查目标与源文件的时间戳，当源文件时间戳更新时，make会更新依赖它的链路上所有目录。例如，当test1.c更新时，再次执行make命令，依次会重新生成test1.o、libtest.a、test。

变量与函数
-----

上面的Makefile中，所有的依赖关系中直接使用文件名称，当新增加文件或者模块时，需要手动去添加新的依赖关系，那么这就比较麻烦，不利于扩展，因此可以使用变量，自动完成新依赖关系添加。

观察上面的依赖关系，可以发现所有的.o中间文件都依赖于一个.c源代码文件。那么这里就可以用变量表达.o中间文件和.c源代码文件，并指定它们之间的依赖关系。Makefile中的变量类型基本上就可以直接理解为字符串类型。

    SRC = test1.c test2.c main.c
    OBJ = test1.o test2.o main.o
    
    ${OBJ}:${SRC}
    	gcc -c ${SRC}
    

与shell脚本中的变量类似，定义变量时，直接使用等号赋值（Makefile中的等号有多种，后面会解释），使用变量时用"${VAR}"表示即可。上面SRC和OBJ变量分别表示.c与.o文件，但是展开写太麻烦了，这里可以用更为方便的办法。

    SRC = $(wildcard *.c)
    OBJ = $(patsubst %.c,%.o,${SRC})
    
    ${OBJ}:${SRC}
    	gcc -c ${SRC}

这里使用了Makefile中的函数，使用"$(<function> <arguments>)"的形式可以调用函数。（这时不打算列举Makefile中支持的函数，因为实在太多了，对于需要用到的函数，可以到网上去查找）。Makefile与shell脚本类似，支持通配符

通配符

作用

\*

匹配0个或者是任意个字符

？

匹配任意一个字符

\[\]

指定匹配的字符放在 "\[\]" 中

"$(wildcard \*.c)"表示将通配符\*.c展开，即"test1.c test2.c main.c"。patsubst是模式替换函数，"$(patsubst %.c,%.o,${SRC})"表示将变量SRC中符合"%.c"形式的字符串，修改为"%.o"形式，例如字符串"main.c"就符合"%.c"形式，%匹配"main"，"main.c"替换之后就变成了"main.o"。因此OBJ变成了"test1.o test2.o main.o"。

上面"${OBJ}:${SRC}"的依赖关系描述其实不是很合适，因为它表示"test1.o test2.o main.o"依赖于"test1.c test2.c main.c"，没有指明哪个.o文件依赖于哪个.c文件，test1.o文件只依赖于test1.c文件，而不是依赖于三个.c文件。那么这里可以用模式匹配的形式来描述依赖关系。

    SRC = $(wildcard *.c)
    OBJ = $(patsubst %.c,%.o,${SRC})
    
    ${OBJ}:%.o:%.c
    	gcc -c $< -o $@
    

"${OBJ}:%.o:%.c"表示OBJ变量中符合"%.o"模式的文件都依赖于"%.c"文件，例如OBJ中的test1.o依赖于test1.c，%在这时就匹配"test1"。另外，这里的命令使用了自动化变量。

自动化变量

作用

$@

规则的目标文件名(依赖关系中冒号:左边的文件，如果a: b c，那么$@指a)

$%

当目标文件是一个静态库文件时，代表静态库的一个成员名。

$<

被依赖文件的第一项(如果a: b c，那么$<指b)

$?

所有比目标文件更新的依赖文件列表，空格分隔

$^

所有依赖文件列表，使用空格分隔(如果a: b c c，那么$^指b c)，不包含重复文件

$+

所有依赖文件列表，使用空格分隔(如果a: b c c，那么$+指b c c)，包含重复文件

$\*

在模式规则和静态模式规则中的"%"所匹配的内容

伪目标
---

    .PHONY : clean
    clean:
    	rm *.o *.a test

这里定义了一个clean目标，它没有依赖。当执行make clean命令时，会删除所有的.o、.a文件和test文件，这样就可以利用Makefile来清除生成的文件。这里make clean并不是为了生成一个名称为clean的文件，为了防止文件同名，可以用.PHONY来声明伪目标。

也可以利用伪目标来一次生成多个目标文件。

    .PHONY : all clean
    all: ${Target1} ${Target2} ${Target3} ...
    
    ${Target1}: ....
    	......
    
    ${Target2}: ....
    	......
    
    ${Target3}: ....
    	......
    
    ......
    
    clean:
    	rm ${Target1} ${Target2} ${Target3} ...

如果按照上面的形式定义目标，执行make all时，会一次生成多个目标文件。

四种等号
----

Makefile中的等号有4种，"="，":="，"?="，"+="。先解释两种比较容易理解的"?="与"+="。

"?="表示，如果左边的变量没有被赋值，那么将等号右边的值赋给左边的变量。下面的例子中，如果VAR\_A被赋过值了，VAR\_A中的值将保持原来的值不变，否则其值变为123。

    VAR_A ?= 123

"+="表示将等号右边的值追加到左边变量中，类似于C语言中的strcat函数。下面的例子中，VAR\_B的最终值为123 456(中间有空格)。

    VAR_B = 123
    VAR_B += 456

"="与":="是比较不好区分的两个等号，可以将"="理解为"址传递"或引用，":="理解为"值传递"。使用"="赋值时，会将整个Makefile展开后再解释被赋值的变量内容，"VAR\_A = ${VAR\_B}"，当后面VAR\_B的值发生改变时VAR\_A的值会跟着进行变化；使用":="赋值时，被赋值的变量的值为此时等号右侧语句表示的值，"VAR\_A := ${VAR\_B}"，如果此时VAR\_B的值是123，那么VAR\_A的值也为123，后面VAR\_B的值被修改了，VAR\_A的值依旧为123。

    var_a = 1 2 3
    var_b = $(var_a)
    var_a += 4
    var_a:
    	echo ${var_a}
    	echo ${var_b}
    ------------------------------------------------------------------------------
    # 执行make var_a的结果:
    echo 1 2 3 4
    1 2 3 4
    echo 1 2 3 4
    1 2 3 4
    ###############################################################################
    var_a = 1 2 3
    var_b := $(var_a)
    var_a += 4
    var_a:
    	echo ${var_a}
    	echo ${var_b}
    -------------------------------------------------------------------------------
    # 执行make var_a的结果:
    echo 1 2 3 4
    1 2 3 4
    echo 1 2 3
    1 2 3
    

在Makefile中是不允许将变量自己的值赋给自己的，Makefile不允许出现循环引用。

    var_a := ${var_a} 1 2 3 # 允许
    var_a = ${var_a} 1 2 3 # 不允许

环境变量
----

Makefile的执行是受shell环境变量影响的，shell环境变量会直接传递到Makefile的执行过程中。

例如，针对语句"VAR\_A ?= yes"，如果在shell中设置过环境变量"export VAR\_A=no"，那么在执行make命令时VAR\_A的值会是no，而不是yes。另外可以在执行make命令时为传递变量的值，如果执行"make VAR\_A=maybe"命令，那么执行过程中VAR\_A是maybe。

利用这个特性，可以在Shell中设置环境变量来影响Makefile的执行过程。同样可以在Makefile中通过修改PATH等变量的值，来解决找不命令的问题。

变量的嵌套使用
-------

Makefile允许变量的嵌套使用，下面的例子中${var\_a}会解释为b，var\_${var\_a}变成var\_b，${var\_${var\_a}}的值就变成了123。

    var_a = b
    var_b = 123
    var_c = ${var_${var_a}}
    var_c:
    	echo ${var_c}
    # 执行make var_c结果
    echo 123
    123

条件判断
----

下面给出了一个条件判断的示例，当DEBUG\_BUILD的值是yes时，CFLAGS中将包含"-ggdb -ggdb3 -gdwarf-2 -D\_DEBUG\_=1 -g"，否则将包含"-O3 -DNDEBUG"，通过这段语句，可以在环境变量中设置DEBUG\_BUILD，是否生成调试版本的程序。

    ifeq (${DEBUG_BUILD},"yes")
    CFLAGS += -ggdb -ggdb3 -gdwarf-2 -D_DEBUG_=1 -g
    else
    CFLAGS += -O3 -DNDEBUG
    endif

关于@与-
-----

在执行make命令时，会打印Makefile里面执行的command，有时候Command过长，不容易查看编译过程中出现的错误与警告，可以通过在command前加上@来取消打印command。

    var_a = 123
    var_a:
    	echo ${var_a}
    -----------------------------
    # 执行make var_a输出
    echo 123
    123
    ##############################
    var_a = 123
    var_a:
    	@echo ${var_a}
    ------------------------------
    # 执行make var_a输出
    123

生成target的过程中，可能需要执行多条命令，执行过程中也可能出现错误， 一般出现错误后，make命令会立即退出，停止编译。如果想要忽略执行过程中的错误，可以在command前加上-来忽略这条命令的执行错误。

    var_a = 123
    var_a:
    	@ls dir1
    	@echo ${var_a}
    ----------------------------------------------
    # 执行make var_a输出
    ls: 无法访问 'dir1': 没有那个文件或目录
    make: [Makefile:19：var_a] 错误 2
    ##############################################
    var_a = 123
    var_a:
    	-@ls dir1
    	@echo ${var_a}
    ----------------------------------------------
    # 执行make var_a输出
    ls: 无法访问 'dir1': 没有那个文件或目录
    make: [Makefile:19：var_a] 错误 2 (已忽略）
    123

关于依赖中的头文件
---------

Makefile与C/C++一样，支持include另外一文件，这个机制允许Makefile可以根据不同的环境或者平台设置不同编译过程。当此文件找不到时，Makefile会报错。

但是这里更想提及的是另外一条语法sinclude的妙用，sinclude在找不到文件时，并不会报错，会直接跳过。利用这个机制，可以更新目标文件的依赖关系。

在上面举过的例子中，所有.o文件仅依赖于一个.c文件，而这个.c文件其实是包含了不少头文件的，所以更加正确的依赖关系应该是下面这样的。

    main.o: main.c include_file1.h include_file2.h include_file3.h ......
        gcc @< -o $@

或者

    main.o: include_file1.h include_file2.h include_file3.h ......
    
    main.o: main.c 
        gcc @< -o $@

只有这样，当头文件被修改时，make命令才会重新编译目标文件，否则make命令无法知道源码其实被更新了。但是一个源代码文件一般会include多个头文件，而头文件往往又会include其它的头文件，如果要手写整个依赖关系，其过程会十分繁琐。

这里会用到编译器的一个功能，通过在编译参数中增加"-MM -MF <dependence\_file>"参数，可以在编译过程中生成一个<dependence\_file>文本文件，说明在编译这个源代码文件时，实际include了哪些文件。那么这里在Makefile中sinclude这个文件，增加.o文件对这些头文件的依赖关系。而当第一次编译时，虽然<dependence\_file>文件不存在，但是所有中间文件都重新生成了，即被更新了，也不会存在报错。后续修改头文件时，make命令也会检查依赖的头文件时间戳是否比目标文件新，从而更新与之相关的所有目标文件。

make命令参数
--------

当直接运行make命令，后面不接target参数时，默认会生成Makefile中的第一个目标。如果要生成指定目标，需要在make命令后面接target名称。

"make <target> VAR\_A=<var_\__a> -j<num>"表示同时产生<num>个进程编译<target>，同时设置Makefile中变量VAR\_A的值为var\_a。如果-j后面不接数字参数，将会为每个目标文件产生一个进程进行编译，如果工程是源文件过多，可能导致进程数量过多而使计算机没有响应，所以直接使用-j参数而后面不接数字是一个不好的操作。

"make -C /build/path -f make1.mak"表示在开始编译前，先将当前目录切换到/build/path路径下，再执行编译，相当于"cd /build/path && make -f make1.mak"。-f参数用于指定要使用的Makefile文件，如果不使用-f参数，则默认使用当前目录下的名称为"Makefile"的文件。注意，这里的make1.mak文件是存放在/build/path目录下的。

简单Makefile模板
------------

这里给出了一个较为简单的Makefile模板，其最终生成目标为可执行程序，如果最终生成目标为库文件，需要进行简单调整。如果你能看懂，那么上面所讲到的要点基本上都理解了，可以直接利用这个模板为小型工程快速搭建一套编译脚本了。利用这套模板生成目标文件，所有中间文件都存放在单独的tmp目录下，防止与源代码文件混在一起，并且可以通过.dep文件自动更新依赖关系。

    ###################################################################################################
    # 编译工具链设置
    PATH := ${PATH}:/your/tool_chain/path
    TOOL_CHAIN = 
    CC = ${TOOL_CHAIN}gcc
    AR = ${TOOL_CHAIN}ar
    
    DEBUG_BUILD ?= yes
    
    # SHOW_COMMAND=yes，显示编译命令
    ifeq (${SHOW_COMMAND}, yes)
    QUIET :=
    else
    QUIET := @
    endif
    
    ###################################################################################################
    # 目录设置
    # 工程根路径
    PROJ_ROOT = $(abspath ../..)
    # 中间文件缓存文件夹
    TMP_PATH = $(abspath .)/tmp
    # 当前路径
    PWD_PATH = $(abspath .)
    
    ###################################################################################################
    # 源文件.c
    SRC := ${PROJ_ROOT}/module1/*.c
    SRC += ${PROJ_ROOT}/module2/*.c
    # 展开*匹配，获取所有源文件完整路径
    SRC := $(wildcard ${SRC})
    
    ###################################################################################################
    # 头文件路径设置
    INCLUDE_PATH += /include/path1
    INCLUDE_PATH += /include/path2
    INCLUDE_PATH += ${PROJ_ROOT}/include/path1
    INCLUDE_PATH += ${PROJ_ROOT}/include/path2
    
    ###################################################################################################
    # 编译宏设定
    DEFINE_SETTINGS := LINUX
    DEFINE_SETTINGS += A72="A72"
    DEFINE_SETTINGS += TARGET_NUM_CORES=1
    DEFINE_SETTINGS += TARGET_ARCH=64
    DEFINE_SETTINGS += ARCH_64
    DEFINE_SETTINGS += ARM
    
    ###################################################################################################
    # 库路径设置
    # 静态库.a文件夹路径
    STATIC_LIB_PATH := ${PROJ_ROOT}/moduleXXX1/lib
    STATIC_LIB_PATH += ${PROJ_ROOT}/moduleXXX2/lib
    # 动态库.so文件夹路径
    DYNAMIC_LIB_PATH := ${PROJ_ROOT}/moduleXXX3/lib
    
    ###################################################################################################
    # 库设置(静态库)
    STATIC_LIB += static_lib1
    STATIC_LIB += static_lib2
    STATIC_LIB += static_lib3
    STATIC_LIB += static_lib4
    # 库设置(动态库)
    DYNAMIC_LIB := stdc++
    DYNAMIC_LIB += m
    DYNAMIC_LIB += rt
    DYNAMIC_LIB += pthread
    
    ###################################################################################################
    # 编译选项
    CFLAGS := -fPIC -Wall -fms-extensions -Wno-write-strings -Wno-format-security
    CFLAGS += -fno-short-enums -Werror
    CFLAGS += -mlittle-endian  -Wno-format-truncation
    ifeq ("${DEBUG_BUILD}","yes")
    CFLAGS += -ggdb -ggdb3 -gdwarf-2 -D_DEBUG_=1 -g
    else
    CFLAGS += -O3 -DNDEBUG
    endif
    
    ###################################################################################################
    # 生成的中间文件.o
    OBJ := $(patsubst ${PROJ_ROOT}/%.c,${TMP_PATH}/%.o,${SRC})
    # 头文件存放路径设置
    INC := $(foreach path,${INCLUDE_PATH},-I${path})
    # 编译宏设置
    DEF := $(foreach macro,${DEFINE_SETTINGS},-D${macro})
    # 库设置
    LIB := -rdynamic -Wl,--cref
    LIB += $(foreach path,${DYNAMIC_LIB_PATH},"-Wl,-rpath-link=${path}")
    LIB += $(foreach path,${STATIC_LIB_PATH},-L${path})
    LIB += -Wl,-Bstatic -Wl,--start-group
    LIB += $(foreach lib,${STATIC_LIB},-l${lib})
    LIB += -Wl,--end-group
    LIB += -Wl,-Bdynamic
    LIB += $(foreach lib,${DYNAMIC_LIB},-l${lib})
    
    # 生成目标
    TARGET := ${PWD_PATH}/demo/demo
    # 生成目标中的详细符号信息文件
    DEP_FILE := -Wl,-Map=${TMP_PATH}/$(notdir ${TARGET}).dep
    
    .PHONY: all clean
    all: ${TARGET}
    
    ${TARGET}:${OBJ}
    	@echo "[Linking $@]"
    	${QUIET}${CC} ${OBJ} ${CFLAGS} ${LIB} -o $@ ${DEP_FILE} >/dev/null
    
    ${TMP_PATH}/%.o:${PROJ_ROOT}/%.c
    	@echo "[Compiling $@]"
    	@mkdir $(dir $@) -p
    	${QUIET}${CC} -c $< -o $@ ${CFLAGS} ${DEF} ${INC} -MMD -MF $(patsubst %.o,%.dep,$@) -MT '$@'
    
    clean:
    	@echo "[cleaning ${TARGET}]"
    	${QUIET}rm -rf ${TARGET}
    	${QUIET}rm -rf ${TMP_PATH}


======================  

知乎转载