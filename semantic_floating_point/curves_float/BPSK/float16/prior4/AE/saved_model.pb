Є╛
∙▌
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
R
Equal
x"T
y"T
z
"	
Ttype"$
incompatible_shape_errorbool(Р
.
Identity

input"T
output"T"	
Ttype
║
If
cond"Tcond
input2Tin
output2Tout"
Tcondtype"
Tin
list(type)("
Tout
list(type)("
then_branchfunc"
else_branchfunc" 
output_shapeslist(shape)
 И
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Н
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
=
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
Е
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	И
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
╛
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
Ў
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12unknown8║Б
Д
gaussian_noise2/stddevVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namegaussian_noise2/stddev
}
*gaussian_noise2/stddev/Read/ReadVariableOpReadVariableOpgaussian_noise2/stddev*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
t
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

: *
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
: *
dtype0
x
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:  *
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
: *
dtype0
x
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

: *
dtype0
p
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:*
dtype0
x
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

: *
dtype0
p
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
: *
dtype0
x
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:  *
dtype0
p
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
: *
dtype0
x
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *
shared_namedense_5/kernel
q
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel*
_output_shapes

: *
dtype0
p
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
i
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
В
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dense/kernel/m
{
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes

: *
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
: *
dtype0
Ж
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes

:  *
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
: *
dtype0
Ж
Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
w
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes
:*
dtype0
Ж
Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_3/bias/m
w
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes
: *
dtype0
Ж
Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m*
_output_shapes

:  *
dtype0
~
Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_4/bias/m
w
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes
: *
dtype0
Ж
Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m*
_output_shapes

: *
dtype0
~
Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
w
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes
:*
dtype0
В
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *$
shared_nameAdam/dense/kernel/v
{
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes

: *
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
: *
dtype0
Ж
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes

:  *
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
: *
dtype0
Ж
Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
w
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes
:*
dtype0
Ж
Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_3/bias/v
w
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes
: *
dtype0
Ж
Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:  *&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v*
_output_shapes

:  *
dtype0
~
Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nameAdam/dense_4/bias/v
w
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes
: *
dtype0
Ж
Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
: *&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v*
_output_shapes

: *
dtype0
~
Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
w
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
▐H
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ЩH
valueПHBМH BЕH
є
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
 
с
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
trainable_variables
regularization_losses
	variables
	keras_api
^

stddev
trainable_variables
regularization_losses
	variables
	keras_api
╘
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
trainable_variables
regularization_losses
	variables
 	keras_api
░
!iter

"beta_1

#beta_2
	$decay
%learning_rate&mК'mЛ(mМ)mН*mО+mП,mР-mС.mТ/mУ0mФ1mХ&vЦ'vЧ(vШ)vЩ*vЪ+vЫ,vЬ-vЭ.vЮ/vЯ0vа1vб
V
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111
 
^
&0
'1
(2
)3
*4
+5
6
,7
-8
.9
/10
011
112
н
trainable_variables

2layers
3metrics
regularization_losses
	variables
4layer_regularization_losses
5layer_metrics
6non_trainable_variables
 
 
h

&kernel
'bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
h

(kernel
)bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
h

*kernel
+bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
R
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
*
&0
'1
(2
)3
*4
+5
 
*
&0
'1
(2
)3
*4
+5
н
trainable_variables

Glayers
Hmetrics
regularization_losses
	variables
Ilayer_regularization_losses
Jlayer_metrics
Knon_trainable_variables
b`
VARIABLE_VALUEgaussian_noise2/stddev6layer_with_weights-1/stddev/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
н
trainable_variables

Llayers
Mmetrics
regularization_losses
	variables
Nlayer_regularization_losses
Olayer_metrics
Pnon_trainable_variables
 
h

,kernel
-bias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
h

.kernel
/bias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
h

0kernel
1bias
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
*
,0
-1
.2
/3
04
15
 
*
,0
-1
.2
/3
04
15
н
trainable_variables

]layers
^metrics
regularization_losses
	variables
_layer_regularization_losses
`layer_metrics
anon_trainable_variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
dense/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_2/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_2/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_3/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_3/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_4/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_4/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_5/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_5/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3

b0
 
 

0

&0
'1
 

&0
'1
н
7trainable_variables

clayers
dmetrics
8regularization_losses
9	variables
elayer_regularization_losses
flayer_metrics
gnon_trainable_variables

(0
)1
 

(0
)1
н
;trainable_variables

hlayers
imetrics
<regularization_losses
=	variables
jlayer_regularization_losses
klayer_metrics
lnon_trainable_variables

*0
+1
 

*0
+1
н
?trainable_variables

mlayers
nmetrics
@regularization_losses
A	variables
olayer_regularization_losses
player_metrics
qnon_trainable_variables
 
 
 
н
Ctrainable_variables

rlayers
smetrics
Dregularization_losses
E	variables
tlayer_regularization_losses
ulayer_metrics
vnon_trainable_variables
#
0
1
2
3
4
 
 
 
 
 
 
 
 

0

,0
-1
 

,0
-1
н
Qtrainable_variables

wlayers
xmetrics
Rregularization_losses
S	variables
ylayer_regularization_losses
zlayer_metrics
{non_trainable_variables

.0
/1
 

.0
/1
о
Utrainable_variables

|layers
}metrics
Vregularization_losses
W	variables
~layer_regularization_losses
layer_metrics
Аnon_trainable_variables

00
11
 

00
11
▓
Ytrainable_variables
Бlayers
Вmetrics
Zregularization_losses
[	variables
 Гlayer_regularization_losses
Дlayer_metrics
Еnon_trainable_variables

0
1
2
3
 
 
 
 
8

Жtotal

Зcount
И	variables
Й	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ж0
З1

И	variables
us
VARIABLE_VALUEAdam/dense/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dense/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_2/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_2/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_3/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_3/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_4/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_4/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_5/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_5/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dense/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_2/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_2/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_3/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_3/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_4/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_4/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_5/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_5/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_3Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
Ч
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasgaussian_noise2/stddevdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         */
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference_signature_wrapper_271658
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
▌
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*gaussian_noise2/stddev/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOpConst*9
Tin2
02.	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *(
f#R!
__inference__traced_save_272590
ш
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamegaussian_noise2/stddev	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biastotalcountAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/v*8
Tin1
/2-*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__traced_restore_272732┴╟
ц
ю
(model_2_gaussian_noise2_cond_true_2707658
4model_2_gaussian_noise2_cond_readvariableop_resource,
(model_2_gaussian_noise2_cond_placeholder)
%model_2_gaussian_noise2_cond_identityИв+model_2/gaussian_noise2/cond/ReadVariableOp╦
+model_2/gaussian_noise2/cond/ReadVariableOpReadVariableOp4model_2_gaussian_noise2_cond_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_2/gaussian_noise2/cond/ReadVariableOpо
0model_2/gaussian_noise2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_2/gaussian_noise2/cond/strided_slice/stack▓
2model_2/gaussian_noise2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_2/gaussian_noise2/cond/strided_slice/stack_1▓
2model_2/gaussian_noise2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_2/gaussian_noise2/cond/strided_slice/stack_2Ш
*model_2/gaussian_noise2/cond/strided_sliceStridedSlice3model_2/gaussian_noise2/cond/ReadVariableOp:value:09model_2/gaussian_noise2/cond/strided_slice/stack:output:0;model_2/gaussian_noise2/cond/strided_slice/stack_1:output:0;model_2/gaussian_noise2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_2/gaussian_noise2/cond/strided_slice▐
%model_2/gaussian_noise2/cond/IdentityIdentity3model_2/gaussian_noise2/cond/strided_slice:output:0,^model_2/gaussian_noise2/cond/ReadVariableOp*
T0*
_output_shapes
: 2'
%model_2/gaussian_noise2/cond/Identity"W
%model_2_gaussian_noise2_cond_identity.model_2/gaussian_noise2/cond/Identity:output:0**
_input_shapes
::         2Z
+model_2/gaussian_noise2/cond/ReadVariableOp+model_2/gaussian_noise2/cond/ReadVariableOp:-)
'
_output_shapes
:         
▐
╣
(__inference_model_1_layer_call_fn_272283

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2712152
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▌
}
(__inference_dense_3_layer_call_fn_272396

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2710682
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
ц
╬
C__inference_model_2_layer_call_and_return_conditional_losses_271588

inputs
model_271558
model_271560
model_271562
model_271564
model_271566
model_271568
gaussian_noise2_271571
model_1_271574
model_1_271576
model_1_271578
model_1_271580
model_1_271582
model_1_271584
identityИв'gaussian_noise2/StatefulPartitionedCallвmodel/StatefulPartitionedCallвmodel_1/StatefulPartitionedCall╚
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_271558model_271560model_271562model_271564model_271566model_271568*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2710382
model/StatefulPartitionedCall▒
'gaussian_noise2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0gaussian_noise2_271571*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_2713472)
'gaussian_noise2/StatefulPartitionedCallД
model_1/StatefulPartitionedCallStatefulPartitionedCall0gaussian_noise2/StatefulPartitionedCall:output:0model_1_271574model_1_271576model_1_271578model_1_271580model_1_271582model_1_271584*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2714062!
model_1/StatefulPartitionedCallш
IdentityIdentity(model_1/StatefulPartitionedCall:output:0(^gaussian_noise2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::2R
'gaussian_noise2/StatefulPartitionedCall'gaussian_noise2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▐
╣
(__inference_model_1_layer_call_fn_272266

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2711792
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
х3
╢
)model_2_gaussian_noise2_cond_false_2707668
4model_2_gaussian_noise2_cond_readvariableop_resourceC
?model_2_gaussian_noise2_cond_shape_model_2_model_lambda_truediv)
%model_2_gaussian_noise2_cond_identityИв+model_2/gaussian_noise2/cond/ReadVariableOpв-model_2/gaussian_noise2/cond/ReadVariableOp_1╖
"model_2/gaussian_noise2/cond/ShapeShape?model_2_gaussian_noise2_cond_shape_model_2_model_lambda_truediv*
T0*
_output_shapes
:2$
"model_2/gaussian_noise2/cond/Shapeо
0model_2/gaussian_noise2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0model_2/gaussian_noise2/cond/strided_slice/stack▓
2model_2/gaussian_noise2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2model_2/gaussian_noise2/cond/strided_slice/stack_1▓
2model_2/gaussian_noise2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2model_2/gaussian_noise2/cond/strided_slice/stack_2Р
*model_2/gaussian_noise2/cond/strided_sliceStridedSlice+model_2/gaussian_noise2/cond/Shape:output:09model_2/gaussian_noise2/cond/strided_slice/stack:output:0;model_2/gaussian_noise2/cond/strided_slice/stack_1:output:0;model_2/gaussian_noise2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*model_2/gaussian_noise2/cond/strided_slice╗
$model_2/gaussian_noise2/cond/Shape_1Shape?model_2_gaussian_noise2_cond_shape_model_2_model_lambda_truediv*
T0*
_output_shapes
:2&
$model_2/gaussian_noise2/cond/Shape_1╦
+model_2/gaussian_noise2/cond/ReadVariableOpReadVariableOp4model_2_gaussian_noise2_cond_readvariableop_resource*
_output_shapes
:*
dtype02-
+model_2/gaussian_noise2/cond/ReadVariableOp▓
2model_2/gaussian_noise2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2model_2/gaussian_noise2/cond/strided_slice_1/stack╢
4model_2/gaussian_noise2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_2/gaussian_noise2/cond/strided_slice_1/stack_1╢
4model_2/gaussian_noise2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_2/gaussian_noise2/cond/strided_slice_1/stack_2в
,model_2/gaussian_noise2/cond/strided_slice_1StridedSlice3model_2/gaussian_noise2/cond/ReadVariableOp:value:0;model_2/gaussian_noise2/cond/strided_slice_1/stack:output:0=model_2/gaussian_noise2/cond/strided_slice_1/stack_1:output:0=model_2/gaussian_noise2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_2/gaussian_noise2/cond/strided_slice_1г
 model_2/gaussian_noise2/cond/LogLog5model_2/gaussian_noise2/cond/strided_slice_1:output:0*
T0*
_output_shapes
: 2"
 model_2/gaussian_noise2/cond/Log╧
-model_2/gaussian_noise2/cond/ReadVariableOp_1ReadVariableOp4model_2_gaussian_noise2_cond_readvariableop_resource*
_output_shapes
:*
dtype02/
-model_2/gaussian_noise2/cond/ReadVariableOp_1▓
2model_2/gaussian_noise2/cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:24
2model_2/gaussian_noise2/cond/strided_slice_2/stack╢
4model_2/gaussian_noise2/cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4model_2/gaussian_noise2/cond/strided_slice_2/stack_1╢
4model_2/gaussian_noise2/cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4model_2/gaussian_noise2/cond/strided_slice_2/stack_2д
,model_2/gaussian_noise2/cond/strided_slice_2StridedSlice5model_2/gaussian_noise2/cond/ReadVariableOp_1:value:0;model_2/gaussian_noise2/cond/strided_slice_2/stack:output:0=model_2/gaussian_noise2/cond/strided_slice_2/stack_1:output:0=model_2/gaussian_noise2/cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,model_2/gaussian_noise2/cond/strided_slice_2з
"model_2/gaussian_noise2/cond/Log_1Log5model_2/gaussian_noise2/cond/strided_slice_2:output:0*
T0*
_output_shapes
: 2$
"model_2/gaussian_noise2/cond/Log_1м
3model_2/gaussian_noise2/cond/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :25
3model_2/gaussian_noise2/cond/random_uniform/shape/1П
1model_2/gaussian_noise2/cond/random_uniform/shapePack3model_2/gaussian_noise2/cond/strided_slice:output:0<model_2/gaussian_noise2/cond/random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:23
1model_2/gaussian_noise2/cond/random_uniform/shapeб
9model_2/gaussian_noise2/cond/random_uniform/RandomUniformRandomUniform:model_2/gaussian_noise2/cond/random_uniform/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2Ь▀м2;
9model_2/gaussian_noise2/cond/random_uniform/RandomUniform╪
/model_2/gaussian_noise2/cond/random_uniform/subSub&model_2/gaussian_noise2/cond/Log_1:y:0$model_2/gaussian_noise2/cond/Log:y:0*
T0*
_output_shapes
: 21
/model_2/gaussian_noise2/cond/random_uniform/subФ
/model_2/gaussian_noise2/cond/random_uniform/mulMulBmodel_2/gaussian_noise2/cond/random_uniform/RandomUniform:output:03model_2/gaussian_noise2/cond/random_uniform/sub:z:0*
T0*'
_output_shapes
:         21
/model_2/gaussian_noise2/cond/random_uniform/mulю
+model_2/gaussian_noise2/cond/random_uniformAdd3model_2/gaussian_noise2/cond/random_uniform/mul:z:0$model_2/gaussian_noise2/cond/Log:y:0*
T0*'
_output_shapes
:         2-
+model_2/gaussian_noise2/cond/random_uniformо
 model_2/gaussian_noise2/cond/ExpExp/model_2/gaussian_noise2/cond/random_uniform:z:0*
T0*'
_output_shapes
:         2"
 model_2/gaussian_noise2/cond/ExpР
%model_2/gaussian_noise2/cond/IdentityIdentity$model_2/gaussian_noise2/cond/Exp:y:0,^model_2/gaussian_noise2/cond/ReadVariableOp.^model_2/gaussian_noise2/cond/ReadVariableOp_1*
T0*'
_output_shapes
:         2'
%model_2/gaussian_noise2/cond/Identity"W
%model_2_gaussian_noise2_cond_identity.model_2/gaussian_noise2/cond/Identity:output:0**
_input_shapes
::         2Z
+model_2/gaussian_noise2/cond/ReadVariableOp+model_2/gaussian_noise2/cond/ReadVariableOp2^
-model_2/gaussian_noise2/cond/ReadVariableOp_1-model_2/gaussian_noise2/cond/ReadVariableOp_1:-)
'
_output_shapes
:         
ц
╬
C__inference_model_2_layer_call_and_return_conditional_losses_271524

inputs
model_271494
model_271496
model_271498
model_271500
model_271502
model_271504
gaussian_noise2_271507
model_1_271510
model_1_271512
model_1_271514
model_1_271516
model_1_271518
model_1_271520
identityИв'gaussian_noise2/StatefulPartitionedCallвmodel/StatefulPartitionedCallвmodel_1/StatefulPartitionedCall╚
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_271494model_271496model_271498model_271500model_271502model_271504*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2710012
model/StatefulPartitionedCall▒
'gaussian_noise2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0gaussian_noise2_271507*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_2713472)
'gaussian_noise2/StatefulPartitionedCallД
model_1/StatefulPartitionedCallStatefulPartitionedCall0gaussian_noise2/StatefulPartitionedCall:output:0model_1_271510model_1_271512model_1_271514model_1_271516model_1_271518model_1_271520*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2713822!
model_1/StatefulPartitionedCallш
IdentityIdentity(model_1/StatefulPartitionedCall:output:0(^gaussian_noise2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::2R
'gaussian_noise2/StatefulPartitionedCall'gaussian_noise2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┌
╖
&__inference_model_layer_call_fn_272046

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2710382
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
с
║
(__inference_model_1_layer_call_fn_271194
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2711792
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
▀
^
B__inference_lambda_layer_call_and_return_conditional_losses_270932

inputs
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y[
powPowinputspow/y:output:0*
T0*'
_output_shapes
:         2
powS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/y^
addAddV2pow:z:0add/y:output:0*
T0*'
_output_shapes
:         2
addr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicesx
MeanMeanadd:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
MeanL
SqrtSqrtMean:output:0*
T0*
_output_shapes

:2
Sqrta
truedivRealDivinputsSqrt:y:0*
T0*'
_output_shapes
:         2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к
┴
C__inference_model_1_layer_call_and_return_conditional_losses_271157
input_2
dense_3_271141
dense_3_271143
dense_4_271146
dense_4_271148
dense_5_271151
dense_5_271153
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallУ
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_3_271141dense_3_271143*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2710682!
dense_3/StatefulPartitionedCall┤
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_271146dense_4_271148*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2710952!
dense_4/StatefulPartitionedCall┤
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_271151dense_5_271153*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2711212!
dense_5/StatefulPartitionedCallт
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
├	
v
cond_true_272061 
cond_readvariableop_resource
cond_placeholder
cond_identityИвcond/ReadVariableOpГ
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp~
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice/stackВ
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1В
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2И
cond/strided_sliceStridedSlicecond/ReadVariableOp:value:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice~
cond/IdentityIdentitycond/strided_slice:output:0^cond/ReadVariableOp*
T0*
_output_shapes
: 2
cond/Identity"'
cond_identitycond/Identity:output:0**
_input_shapes
::         2*
cond/ReadVariableOpcond/ReadVariableOp:-)
'
_output_shapes
:         
ы	
┌
A__inference_dense_layer_call_and_return_conditional_losses_272294

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
э	
▄
C__inference_dense_1_layer_call_and_return_conditional_losses_272314

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
ы	
┌
A__inference_dense_layer_call_and_return_conditional_losses_270851

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╒-
■
!gaussian_noise2_cond_false_2717020
,gaussian_noise2_cond_readvariableop_resource3
/gaussian_noise2_cond_shape_model_lambda_truediv!
gaussian_noise2_cond_identityИв#gaussian_noise2/cond/ReadVariableOpв%gaussian_noise2/cond/ReadVariableOp_1Ч
gaussian_noise2/cond/ShapeShape/gaussian_noise2_cond_shape_model_lambda_truediv*
T0*
_output_shapes
:2
gaussian_noise2/cond/ShapeЮ
(gaussian_noise2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gaussian_noise2/cond/strided_slice/stackв
*gaussian_noise2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*gaussian_noise2/cond/strided_slice/stack_1в
*gaussian_noise2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gaussian_noise2/cond/strided_slice/stack_2р
"gaussian_noise2/cond/strided_sliceStridedSlice#gaussian_noise2/cond/Shape:output:01gaussian_noise2/cond/strided_slice/stack:output:03gaussian_noise2/cond/strided_slice/stack_1:output:03gaussian_noise2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"gaussian_noise2/cond/strided_sliceЫ
gaussian_noise2/cond/Shape_1Shape/gaussian_noise2_cond_shape_model_lambda_truediv*
T0*
_output_shapes
:2
gaussian_noise2/cond/Shape_1│
#gaussian_noise2/cond/ReadVariableOpReadVariableOp,gaussian_noise2_cond_readvariableop_resource*
_output_shapes
:*
dtype02%
#gaussian_noise2/cond/ReadVariableOpв
*gaussian_noise2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*gaussian_noise2/cond/strided_slice_1/stackж
,gaussian_noise2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,gaussian_noise2/cond/strided_slice_1/stack_1ж
,gaussian_noise2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gaussian_noise2/cond/strided_slice_1/stack_2Є
$gaussian_noise2/cond/strided_slice_1StridedSlice+gaussian_noise2/cond/ReadVariableOp:value:03gaussian_noise2/cond/strided_slice_1/stack:output:05gaussian_noise2/cond/strided_slice_1/stack_1:output:05gaussian_noise2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$gaussian_noise2/cond/strided_slice_1Л
gaussian_noise2/cond/LogLog-gaussian_noise2/cond/strided_slice_1:output:0*
T0*
_output_shapes
: 2
gaussian_noise2/cond/Log╖
%gaussian_noise2/cond/ReadVariableOp_1ReadVariableOp,gaussian_noise2_cond_readvariableop_resource*
_output_shapes
:*
dtype02'
%gaussian_noise2/cond/ReadVariableOp_1в
*gaussian_noise2/cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*gaussian_noise2/cond/strided_slice_2/stackж
,gaussian_noise2/cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,gaussian_noise2/cond/strided_slice_2/stack_1ж
,gaussian_noise2/cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gaussian_noise2/cond/strided_slice_2/stack_2Ї
$gaussian_noise2/cond/strided_slice_2StridedSlice-gaussian_noise2/cond/ReadVariableOp_1:value:03gaussian_noise2/cond/strided_slice_2/stack:output:05gaussian_noise2/cond/strided_slice_2/stack_1:output:05gaussian_noise2/cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$gaussian_noise2/cond/strided_slice_2П
gaussian_noise2/cond/Log_1Log-gaussian_noise2/cond/strided_slice_2:output:0*
T0*
_output_shapes
: 2
gaussian_noise2/cond/Log_1Ь
+gaussian_noise2/cond/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+gaussian_noise2/cond/random_uniform/shape/1я
)gaussian_noise2/cond/random_uniform/shapePack+gaussian_noise2/cond/strided_slice:output:04gaussian_noise2/cond/random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2+
)gaussian_noise2/cond/random_uniform/shapeЙ
1gaussian_noise2/cond/random_uniform/RandomUniformRandomUniform2gaussian_noise2/cond/random_uniform/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2вЙ▐23
1gaussian_noise2/cond/random_uniform/RandomUniform╕
'gaussian_noise2/cond/random_uniform/subSubgaussian_noise2/cond/Log_1:y:0gaussian_noise2/cond/Log:y:0*
T0*
_output_shapes
: 2)
'gaussian_noise2/cond/random_uniform/subЇ
'gaussian_noise2/cond/random_uniform/mulMul:gaussian_noise2/cond/random_uniform/RandomUniform:output:0+gaussian_noise2/cond/random_uniform/sub:z:0*
T0*'
_output_shapes
:         2)
'gaussian_noise2/cond/random_uniform/mul╬
#gaussian_noise2/cond/random_uniformAdd+gaussian_noise2/cond/random_uniform/mul:z:0gaussian_noise2/cond/Log:y:0*
T0*'
_output_shapes
:         2%
#gaussian_noise2/cond/random_uniformЦ
gaussian_noise2/cond/ExpExp'gaussian_noise2/cond/random_uniform:z:0*
T0*'
_output_shapes
:         2
gaussian_noise2/cond/Expш
gaussian_noise2/cond/IdentityIdentitygaussian_noise2/cond/Exp:y:0$^gaussian_noise2/cond/ReadVariableOp&^gaussian_noise2/cond/ReadVariableOp_1*
T0*'
_output_shapes
:         2
gaussian_noise2/cond/Identity"G
gaussian_noise2_cond_identity&gaussian_noise2/cond/Identity:output:0**
_input_shapes
::         2J
#gaussian_noise2/cond/ReadVariableOp#gaussian_noise2/cond/ReadVariableOp2N
%gaussian_noise2/cond/ReadVariableOp_1%gaussian_noise2/cond/ReadVariableOp_1:-)
'
_output_shapes
:         
э	
▄
C__inference_dense_1_layer_call_and_return_conditional_losses_270878

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
э	
▄
C__inference_dense_3_layer_call_and_return_conditional_losses_272387

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ж
╞
 gaussian_noise2_cond_true_2718150
,gaussian_noise2_cond_readvariableop_resource$
 gaussian_noise2_cond_placeholder!
gaussian_noise2_cond_identityИв#gaussian_noise2/cond/ReadVariableOp│
#gaussian_noise2/cond/ReadVariableOpReadVariableOp,gaussian_noise2_cond_readvariableop_resource*
_output_shapes
:*
dtype02%
#gaussian_noise2/cond/ReadVariableOpЮ
(gaussian_noise2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gaussian_noise2/cond/strided_slice/stackв
*gaussian_noise2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*gaussian_noise2/cond/strided_slice/stack_1в
*gaussian_noise2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gaussian_noise2/cond/strided_slice/stack_2ш
"gaussian_noise2/cond/strided_sliceStridedSlice+gaussian_noise2/cond/ReadVariableOp:value:01gaussian_noise2/cond/strided_slice/stack:output:03gaussian_noise2/cond/strided_slice/stack_1:output:03gaussian_noise2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"gaussian_noise2/cond/strided_slice╛
gaussian_noise2/cond/IdentityIdentity+gaussian_noise2/cond/strided_slice:output:0$^gaussian_noise2/cond/ReadVariableOp*
T0*
_output_shapes
: 2
gaussian_noise2/cond/Identity"G
gaussian_noise2_cond_identity&gaussian_noise2/cond/Identity:output:0**
_input_shapes
::         2J
#gaussian_noise2/cond/ReadVariableOp#gaussian_noise2/cond/ReadVariableOp:-)
'
_output_shapes
:         
С	
▄
C__inference_dense_2_layer_call_and_return_conditional_losses_272333

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
·
╕
A__inference_model_layer_call_and_return_conditional_losses_271001

inputs
dense_270984
dense_270986
dense_1_270989
dense_1_270991
dense_2_270994
dense_2_270996
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallИ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_270984dense_270986*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2708512
dense/StatefulPartitionedCall▓
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_270989dense_1_270991*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2708782!
dense_1/StatefulPartitionedCall┤
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_270994dense_2_270996*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2709042!
dense_2/StatefulPartitionedCallє
lambda/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2709322
lambda/PartitionedCall╫
IdentityIdentitylambda/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
└
╣
(__inference_model_1_layer_call_fn_272184

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2713822
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:::::::22
StatefulPartitionedCallStatefulPartitionedCall:@ <

_output_shapes
:
 
_user_specified_nameinputs
к
┴
C__inference_model_1_layer_call_and_return_conditional_losses_271138
input_2
dense_3_271079
dense_3_271081
dense_4_271106
dense_4_271108
dense_5_271132
dense_5_271134
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallУ
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_3_271079dense_3_271081*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2710682!
dense_3/StatefulPartitionedCall┤
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_271106dense_4_271108*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2710952!
dense_4/StatefulPartitionedCall┤
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_271132dense_5_271134*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2711212!
dense_5/StatefulPartitionedCallт
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
▌
╕
&__inference_model_layer_call_fn_271016
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2710012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
▌z
х
!__inference__wrapped_model_270836
input_36
2model_2_model_dense_matmul_readvariableop_resource7
3model_2_model_dense_biasadd_readvariableop_resource8
4model_2_model_dense_1_matmul_readvariableop_resource9
5model_2_model_dense_1_biasadd_readvariableop_resource8
4model_2_model_dense_2_matmul_readvariableop_resource9
5model_2_model_dense_2_biasadd_readvariableop_resource3
/model_2_gaussian_noise2_readvariableop_resource:
6model_2_model_1_dense_3_matmul_readvariableop_resource;
7model_2_model_1_dense_3_biasadd_readvariableop_resource:
6model_2_model_1_dense_4_matmul_readvariableop_resource;
7model_2_model_1_dense_4_biasadd_readvariableop_resource:
6model_2_model_1_dense_5_matmul_readvariableop_resource;
7model_2_model_1_dense_5_biasadd_readvariableop_resource
identityИв&model_2/gaussian_noise2/ReadVariableOpв(model_2/gaussian_noise2/ReadVariableOp_1вmodel_2/gaussian_noise2/condв*model_2/model/dense/BiasAdd/ReadVariableOpв)model_2/model/dense/MatMul/ReadVariableOpв,model_2/model/dense_1/BiasAdd/ReadVariableOpв+model_2/model/dense_1/MatMul/ReadVariableOpв,model_2/model/dense_2/BiasAdd/ReadVariableOpв+model_2/model/dense_2/MatMul/ReadVariableOpв.model_2/model_1/dense_3/BiasAdd/ReadVariableOpв-model_2/model_1/dense_3/MatMul/ReadVariableOpв.model_2/model_1/dense_4/BiasAdd/ReadVariableOpв-model_2/model_1/dense_4/MatMul/ReadVariableOpв.model_2/model_1/dense_5/BiasAdd/ReadVariableOpв-model_2/model_1/dense_5/MatMul/ReadVariableOp╔
)model_2/model/dense/MatMul/ReadVariableOpReadVariableOp2model_2_model_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02+
)model_2/model/dense/MatMul/ReadVariableOp░
model_2/model/dense/MatMulMatMulinput_31model_2/model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_2/model/dense/MatMul╚
*model_2/model/dense/BiasAdd/ReadVariableOpReadVariableOp3model_2_model_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*model_2/model/dense/BiasAdd/ReadVariableOp╤
model_2/model/dense/BiasAddBiasAdd$model_2/model/dense/MatMul:product:02model_2/model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_2/model/dense/BiasAddФ
model_2/model/dense/ReluRelu$model_2/model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model_2/model/dense/Relu╧
+model_2/model/dense_1/MatMul/ReadVariableOpReadVariableOp4model_2_model_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+model_2/model/dense_1/MatMul/ReadVariableOp╒
model_2/model/dense_1/MatMulMatMul&model_2/model/dense/Relu:activations:03model_2/model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_2/model/dense_1/MatMul╬
,model_2/model/dense_1/BiasAdd/ReadVariableOpReadVariableOp5model_2_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_2/model/dense_1/BiasAdd/ReadVariableOp┘
model_2/model/dense_1/BiasAddBiasAdd&model_2/model/dense_1/MatMul:product:04model_2/model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_2/model/dense_1/BiasAddЪ
model_2/model/dense_1/ReluRelu&model_2/model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model_2/model/dense_1/Relu╧
+model_2/model/dense_2/MatMul/ReadVariableOpReadVariableOp4model_2_model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+model_2/model/dense_2/MatMul/ReadVariableOp╫
model_2/model/dense_2/MatMulMatMul(model_2/model/dense_1/Relu:activations:03model_2/model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/model/dense_2/MatMul╬
,model_2/model/dense_2/BiasAdd/ReadVariableOpReadVariableOp5model_2_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_2/model/dense_2/BiasAdd/ReadVariableOp┘
model_2/model/dense_2/BiasAddBiasAdd&model_2/model/dense_2/MatMul:product:04model_2/model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_2/model/dense_2/BiasAdd}
model_2/model/lambda/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_2/model/lambda/pow/y║
model_2/model/lambda/powPow&model_2/model/dense_2/BiasAdd:output:0#model_2/model/lambda/pow/y:output:0*
T0*'
_output_shapes
:         2
model_2/model/lambda/pow}
model_2/model/lambda/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_2/model/lambda/add/y▓
model_2/model/lambda/addAddV2model_2/model/lambda/pow:z:0#model_2/model/lambda/add/y:output:0*
T0*'
_output_shapes
:         2
model_2/model/lambda/addЬ
+model_2/model/lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_2/model/lambda/Mean/reduction_indices╠
model_2/model/lambda/MeanMeanmodel_2/model/lambda/add:z:04model_2/model/lambda/Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
model_2/model/lambda/MeanЛ
model_2/model/lambda/SqrtSqrt"model_2/model/lambda/Mean:output:0*
T0*
_output_shapes

:2
model_2/model/lambda/Sqrt└
model_2/model/lambda/truedivRealDiv&model_2/model/dense_2/BiasAdd:output:0model_2/model/lambda/Sqrt:y:0*
T0*'
_output_shapes
:         2
model_2/model/lambda/truediv╝
&model_2/gaussian_noise2/ReadVariableOpReadVariableOp/model_2_gaussian_noise2_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_2/gaussian_noise2/ReadVariableOpд
+model_2/gaussian_noise2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+model_2/gaussian_noise2/strided_slice/stackи
-model_2/gaussian_noise2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_2/gaussian_noise2/strided_slice/stack_1и
-model_2/gaussian_noise2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-model_2/gaussian_noise2/strided_slice/stack_2·
%model_2/gaussian_noise2/strided_sliceStridedSlice.model_2/gaussian_noise2/ReadVariableOp:value:04model_2/gaussian_noise2/strided_slice/stack:output:06model_2/gaussian_noise2/strided_slice/stack_1:output:06model_2/gaussian_noise2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%model_2/gaussian_noise2/strided_slice└
(model_2/gaussian_noise2/ReadVariableOp_1ReadVariableOp/model_2_gaussian_noise2_readvariableop_resource*
_output_shapes
:*
dtype02*
(model_2/gaussian_noise2/ReadVariableOp_1и
-model_2/gaussian_noise2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2/
-model_2/gaussian_noise2/strided_slice_1/stackм
/model_2/gaussian_noise2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/model_2/gaussian_noise2/strided_slice_1/stack_1м
/model_2/gaussian_noise2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/model_2/gaussian_noise2/strided_slice_1/stack_2Ж
'model_2/gaussian_noise2/strided_slice_1StridedSlice0model_2/gaussian_noise2/ReadVariableOp_1:value:06model_2/gaussian_noise2/strided_slice_1/stack:output:08model_2/gaussian_noise2/strided_slice_1/stack_1:output:08model_2/gaussian_noise2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'model_2/gaussian_noise2/strided_slice_1ъ
model_2/gaussian_noise2/EqualEqual.model_2/gaussian_noise2/strided_slice:output:00model_2/gaussian_noise2/strided_slice_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
model_2/gaussian_noise2/Equal╫
model_2/gaussian_noise2/condIf!model_2/gaussian_noise2/Equal:z:0/model_2_gaussian_noise2_readvariableop_resource model_2/model/lambda/truediv:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
:*#
_read_only_resource_inputs
*<
else_branch-R+
)model_2_gaussian_noise2_cond_false_270766*
output_shapes
:*;
then_branch,R*
(model_2_gaussian_noise2_cond_true_2707652
model_2/gaussian_noise2/condд
%model_2/gaussian_noise2/cond/IdentityIdentity%model_2/gaussian_noise2/cond:output:0*
T0*
_output_shapes
:2'
%model_2/gaussian_noise2/cond/IdentityО
model_2/gaussian_noise2/ShapeShape model_2/model/lambda/truediv:z:0*
T0*
_output_shapes
:2
model_2/gaussian_noise2/ShapeЭ
*model_2/gaussian_noise2/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*model_2/gaussian_noise2/random_normal/meanб
,model_2/gaussian_noise2/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2.
,model_2/gaussian_noise2/random_normal/stddevЦ
:model_2/gaussian_noise2/random_normal/RandomStandardNormalRandomStandardNormal&model_2/gaussian_noise2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2╔╒ 2<
:model_2/gaussian_noise2/random_normal/RandomStandardNormalЛ
)model_2/gaussian_noise2/random_normal/mulMulCmodel_2/gaussian_noise2/random_normal/RandomStandardNormal:output:05model_2/gaussian_noise2/random_normal/stddev:output:0*
T0*'
_output_shapes
:         2+
)model_2/gaussian_noise2/random_normal/mulы
%model_2/gaussian_noise2/random_normalAdd-model_2/gaussian_noise2/random_normal/mul:z:03model_2/gaussian_noise2/random_normal/mean:output:0*
T0*'
_output_shapes
:         2'
%model_2/gaussian_noise2/random_normal┐
model_2/gaussian_noise2/mulMul.model_2/gaussian_noise2/cond/Identity:output:0)model_2/gaussian_noise2/random_normal:z:0*
T0*
_output_shapes
:2
model_2/gaussian_noise2/mul╕
model_2/gaussian_noise2/addAddV2 model_2/model/lambda/truediv:z:0model_2/gaussian_noise2/mul:z:0*
T0*'
_output_shapes
:         2
model_2/gaussian_noise2/add╒
-model_2/model_1/dense_3/MatMul/ReadVariableOpReadVariableOp6model_2_model_1_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02/
-model_2/model_1/dense_3/MatMul/ReadVariableOp╘
model_2/model_1/dense_3/MatMulMatMulmodel_2/gaussian_noise2/add:z:05model_2/model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2 
model_2/model_1/dense_3/MatMul╘
.model_2/model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp7model_2_model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.model_2/model_1/dense_3/BiasAdd/ReadVariableOpс
model_2/model_1/dense_3/BiasAddBiasAdd(model_2/model_1/dense_3/MatMul:product:06model_2/model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2!
model_2/model_1/dense_3/BiasAddа
model_2/model_1/dense_3/ReluRelu(model_2/model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model_2/model_1/dense_3/Relu╒
-model_2/model_1/dense_4/MatMul/ReadVariableOpReadVariableOp6model_2_model_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02/
-model_2/model_1/dense_4/MatMul/ReadVariableOp▀
model_2/model_1/dense_4/MatMulMatMul*model_2/model_1/dense_3/Relu:activations:05model_2/model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2 
model_2/model_1/dense_4/MatMul╘
.model_2/model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp7model_2_model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.model_2/model_1/dense_4/BiasAdd/ReadVariableOpс
model_2/model_1/dense_4/BiasAddBiasAdd(model_2/model_1/dense_4/MatMul:product:06model_2/model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2!
model_2/model_1/dense_4/BiasAddа
model_2/model_1/dense_4/ReluRelu(model_2/model_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model_2/model_1/dense_4/Relu╒
-model_2/model_1/dense_5/MatMul/ReadVariableOpReadVariableOp6model_2_model_1_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02/
-model_2/model_1/dense_5/MatMul/ReadVariableOp▀
model_2/model_1/dense_5/MatMulMatMul*model_2/model_1/dense_4/Relu:activations:05model_2/model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2 
model_2/model_1/dense_5/MatMul╘
.model_2/model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp7model_2_model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.model_2/model_1/dense_5/BiasAdd/ReadVariableOpс
model_2/model_1/dense_5/BiasAddBiasAdd(model_2/model_1/dense_5/MatMul:product:06model_2/model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2!
model_2/model_1/dense_5/BiasAddе
IdentityIdentity(model_2/model_1/dense_5/BiasAdd:output:0'^model_2/gaussian_noise2/ReadVariableOp)^model_2/gaussian_noise2/ReadVariableOp_1^model_2/gaussian_noise2/cond+^model_2/model/dense/BiasAdd/ReadVariableOp*^model_2/model/dense/MatMul/ReadVariableOp-^model_2/model/dense_1/BiasAdd/ReadVariableOp,^model_2/model/dense_1/MatMul/ReadVariableOp-^model_2/model/dense_2/BiasAdd/ReadVariableOp,^model_2/model/dense_2/MatMul/ReadVariableOp/^model_2/model_1/dense_3/BiasAdd/ReadVariableOp.^model_2/model_1/dense_3/MatMul/ReadVariableOp/^model_2/model_1/dense_4/BiasAdd/ReadVariableOp.^model_2/model_1/dense_4/MatMul/ReadVariableOp/^model_2/model_1/dense_5/BiasAdd/ReadVariableOp.^model_2/model_1/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::2P
&model_2/gaussian_noise2/ReadVariableOp&model_2/gaussian_noise2/ReadVariableOp2T
(model_2/gaussian_noise2/ReadVariableOp_1(model_2/gaussian_noise2/ReadVariableOp_12<
model_2/gaussian_noise2/condmodel_2/gaussian_noise2/cond2X
*model_2/model/dense/BiasAdd/ReadVariableOp*model_2/model/dense/BiasAdd/ReadVariableOp2V
)model_2/model/dense/MatMul/ReadVariableOp)model_2/model/dense/MatMul/ReadVariableOp2\
,model_2/model/dense_1/BiasAdd/ReadVariableOp,model_2/model/dense_1/BiasAdd/ReadVariableOp2Z
+model_2/model/dense_1/MatMul/ReadVariableOp+model_2/model/dense_1/MatMul/ReadVariableOp2\
,model_2/model/dense_2/BiasAdd/ReadVariableOp,model_2/model/dense_2/BiasAdd/ReadVariableOp2Z
+model_2/model/dense_2/MatMul/ReadVariableOp+model_2/model/dense_2/MatMul/ReadVariableOp2`
.model_2/model_1/dense_3/BiasAdd/ReadVariableOp.model_2/model_1/dense_3/BiasAdd/ReadVariableOp2^
-model_2/model_1/dense_3/MatMul/ReadVariableOp-model_2/model_1/dense_3/MatMul/ReadVariableOp2`
.model_2/model_1/dense_4/BiasAdd/ReadVariableOp.model_2/model_1/dense_4/BiasAdd/ReadVariableOp2^
-model_2/model_1/dense_4/MatMul/ReadVariableOp-model_2/model_1/dense_4/MatMul/ReadVariableOp2`
.model_2/model_1/dense_5/BiasAdd/ReadVariableOp.model_2/model_1/dense_5/BiasAdd/ReadVariableOp2^
-model_2/model_1/dense_5/MatMul/ReadVariableOp-model_2/model_1/dense_5/MatMul/ReadVariableOp:P L
'
_output_shapes
:         
!
_user_specified_name	input_3
▌
╕
&__inference_model_layer_call_fn_271053
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2710382
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╨
▓
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_272112

inputs
readvariableop_resource
identityИвReadVariableOpвReadVariableOp_1вcondt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpt
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ъ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2Ў
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1К
EqualEqualstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equalн
condIf	Equal:z:0readvariableop_resourceinputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
:*#
_read_only_resource_inputs
*$
else_branchR
cond_false_272062*
output_shapes
:*#
then_branchR
cond_true_2720612
cond\
cond/IdentityIdentitycond:output:0*
T0*
_output_shapes
:2
cond/IdentityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
random_normal/stddev╬
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2н░■2$
"random_normal/RandomStandardNormalл
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         2
random_normal/mulЛ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         2
random_normal_
mulMulcond/Identity:output:0random_normal:z:0*
T0*
_output_shapes
:2
mulG
addAddV2inputsmul:z:0*
T0*
_output_shapes
:2
addw
IdentityIdentityadd:z:0^ReadVariableOp^ReadVariableOp_1^cond*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0**
_input_shapes
:         :2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12
condcond:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ы
░
C__inference_model_1_layer_call_and_return_conditional_losses_272143

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpе
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOpЛ
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_3/Reluе
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOpЯ
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_4/Reluе
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddп
IdentityIdentitydense_5/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:@ <

_output_shapes
:
 
_user_specified_nameinputs
С	
▄
C__inference_dense_5_layer_call_and_return_conditional_losses_272426

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
С
C
'__inference_lambda_layer_call_fn_272371

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2709322
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
з
└
C__inference_model_1_layer_call_and_return_conditional_losses_271179

inputs
dense_3_271163
dense_3_271165
dense_4_271168
dense_4_271170
dense_5_271173
dense_5_271175
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallТ
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_271163dense_3_271165*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2710682!
dense_3/StatefulPartitionedCall┤
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_271168dense_4_271170*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2710952!
dense_4/StatefulPartitionedCall┤
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_271173dense_5_271175*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2711212!
dense_5/StatefulPartitionedCallт
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
з
└
C__inference_model_1_layer_call_and_return_conditional_losses_271215

inputs
dense_3_271199
dense_3_271201
dense_4_271204
dense_4_271206
dense_5_271209
dense_5_271211
identityИвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallвdense_5/StatefulPartitionedCallТ
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_271199dense_3_271201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2710682!
dense_3/StatefulPartitionedCall┤
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_271204dense_4_271206*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2710952!
dense_4/StatefulPartitionedCall┤
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_271209dense_5_271211*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2711212!
dense_5/StatefulPartitionedCallт
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Я	
б
$__inference_signature_wrapper_271658
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityИвStatefulPartitionedCallц
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         */
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__wrapped_model_2708362
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_3
¤
╣
A__inference_model_layer_call_and_return_conditional_losses_270978
input_1
dense_270961
dense_270963
dense_1_270966
dense_1_270968
dense_2_270971
dense_2_270973
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallЙ
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_270961dense_270963*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2708512
dense/StatefulPartitionedCall▓
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_270966dense_1_270968*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2708782!
dense_1/StatefulPartitionedCall┤
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_270971dense_2_270973*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2709042!
dense_2/StatefulPartitionedCallє
lambda/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2709442
lambda/PartitionedCall╫
IdentityIdentitylambda/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
╨
▓
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_271347

inputs
readvariableop_resource
identityИвReadVariableOpвReadVariableOp_1вcondt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpt
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2ъ
strided_sliceStridedSliceReadVariableOp:value:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1x
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2Ў
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1К
EqualEqualstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equalн
condIf	Equal:z:0readvariableop_resourceinputs*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
:*#
_read_only_resource_inputs
*$
else_branchR
cond_false_271297*
output_shapes
:*#
then_branchR
cond_true_2712962
cond\
cond/IdentityIdentitycond:output:0*
T0*
_output_shapes
:2
cond/IdentityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapem
random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
random_normal/meanq
random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2
random_normal/stddev╬
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2лхи2$
"random_normal/RandomStandardNormalл
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:         2
random_normal/mulЛ
random_normalAddrandom_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:         2
random_normal_
mulMulcond/Identity:output:0random_normal:z:0*
T0*
_output_shapes
:2
mulG
addAddV2inputsmul:z:0*
T0*
_output_shapes
:2
addw
IdentityIdentityadd:z:0^ReadVariableOp^ReadVariableOp_1^cond*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0**
_input_shapes
:         :2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12
condcond:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▌
}
(__inference_dense_1_layer_call_fn_272323

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2708782
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┬	
д
(__inference_model_2_layer_call_fn_271948

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         */
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2715882
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
к
░
C__inference_model_1_layer_call_and_return_conditional_losses_272225

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpе
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOpЛ
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_3/Reluе
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOpЯ
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_4/Reluе
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddп
IdentityIdentitydense_5/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
¤
╣
A__inference_model_layer_call_and_return_conditional_losses_270958
input_1
dense_270862
dense_270864
dense_1_270889
dense_1_270891
dense_2_270915
dense_2_270917
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallЙ
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_270862dense_270864*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2708512
dense/StatefulPartitionedCall▓
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_270889dense_1_270891*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2708782!
dense_1/StatefulPartitionedCall┤
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_270915dense_2_270917*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2709042!
dense_2/StatefulPartitionedCallє
lambda/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2709322
lambda/PartitionedCall╫
IdentityIdentitylambda/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_1
▀
^
B__inference_lambda_layer_call_and_return_conditional_losses_272366

inputs
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y[
powPowinputspow/y:output:0*
T0*'
_output_shapes
:         2
powS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/y^
addAddV2pow:z:0add/y:output:0*
T0*'
_output_shapes
:         2
addr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicesx
MeanMeanadd:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
MeanL
SqrtSqrtMean:output:0*
T0*
_output_shapes

:2
Sqrta
truedivRealDivinputsSqrt:y:0*
T0*'
_output_shapes
:         2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
│!
Р
cond_false_272062 
cond_readvariableop_resource
cond_shape_inputs
cond_identityИвcond/ReadVariableOpвcond/ReadVariableOp_1Y

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:2

cond/Shape~
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice/stackВ
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1В
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2А
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice]
cond/Shape_1Shapecond_shape_inputs*
T0*
_output_shapes
:2
cond/Shape_1Г
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOpВ
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stackЖ
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_1/stack_1Ж
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_1/stack_2Т
cond/strided_slice_1StridedSlicecond/ReadVariableOp:value:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_1[
cond/LogLogcond/strided_slice_1:output:0*
T0*
_output_shapes
: 2

cond/LogЗ
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp_1В
cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stackЖ
cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack_1Ж
cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack_2Ф
cond/strided_slice_2StridedSlicecond/ReadVariableOp_1:value:0#cond/strided_slice_2/stack:output:0%cond/strided_slice_2/stack_1:output:0%cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_2_

cond/Log_1Logcond/strided_slice_2:output:0*
T0*
_output_shapes
: 2

cond/Log_1|
cond/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
cond/random_uniform/shape/1п
cond/random_uniform/shapePackcond/strided_slice:output:0$cond/random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2
cond/random_uniform/shape┘
!cond/random_uniform/RandomUniformRandomUniform"cond/random_uniform/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2╕П╩2#
!cond/random_uniform/RandomUniformx
cond/random_uniform/subSubcond/Log_1:y:0cond/Log:y:0*
T0*
_output_shapes
: 2
cond/random_uniform/sub┤
cond/random_uniform/mulMul*cond/random_uniform/RandomUniform:output:0cond/random_uniform/sub:z:0*
T0*'
_output_shapes
:         2
cond/random_uniform/mulО
cond/random_uniformAddcond/random_uniform/mul:z:0cond/Log:y:0*
T0*'
_output_shapes
:         2
cond/random_uniformf
cond/ExpExpcond/random_uniform:z:0*
T0*'
_output_shapes
:         2

cond/ExpШ
cond/IdentityIdentitycond/Exp:y:0^cond/ReadVariableOp^cond/ReadVariableOp_1*
T0*'
_output_shapes
:         2
cond/Identity"'
cond_identitycond/Identity:output:0**
_input_shapes
::         2*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:-)
'
_output_shapes
:         
▀
^
B__inference_lambda_layer_call_and_return_conditional_losses_270944

inputs
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y[
powPowinputspow/y:output:0*
T0*'
_output_shapes
:         2
powS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/y^
addAddV2pow:z:0add/y:output:0*
T0*'
_output_shapes
:         2
addr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicesx
MeanMeanadd:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
MeanL
SqrtSqrtMean:output:0*
T0*
_output_shapes

:2
Sqrta
truedivRealDivinputsSqrt:y:0*
T0*'
_output_shapes
:         2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ы
░
C__inference_model_1_layer_call_and_return_conditional_losses_271406

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpе
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOpЛ
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_3/Reluе
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOpЯ
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_4/Reluе
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddп
IdentityIdentitydense_5/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:@ <

_output_shapes
:
 
_user_specified_nameinputs
щ
╧
C__inference_model_2_layer_call_and_return_conditional_losses_271488
input_3
model_271458
model_271460
model_271462
model_271464
model_271466
model_271468
gaussian_noise2_271471
model_1_271474
model_1_271476
model_1_271478
model_1_271480
model_1_271482
model_1_271484
identityИв'gaussian_noise2/StatefulPartitionedCallвmodel/StatefulPartitionedCallвmodel_1/StatefulPartitionedCall╔
model/StatefulPartitionedCallStatefulPartitionedCallinput_3model_271458model_271460model_271462model_271464model_271466model_271468*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2710382
model/StatefulPartitionedCall▒
'gaussian_noise2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0gaussian_noise2_271471*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_2713472)
'gaussian_noise2/StatefulPartitionedCallД
model_1/StatefulPartitionedCallStatefulPartitionedCall0gaussian_noise2/StatefulPartitionedCall:output:0model_1_271474model_1_271476model_1_271478model_1_271480model_1_271482model_1_271484*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2714062!
model_1/StatefulPartitionedCallш
IdentityIdentity(model_1/StatefulPartitionedCall:output:0(^gaussian_noise2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::2R
'gaussian_noise2/StatefulPartitionedCall'gaussian_noise2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_3
╕j
ж

C__inference_model_2_layer_call_and_return_conditional_losses_271772

inputs.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource+
'gaussian_noise2_readvariableop_resource2
.model_1_dense_3_matmul_readvariableop_resource3
/model_1_dense_3_biasadd_readvariableop_resource2
.model_1_dense_4_matmul_readvariableop_resource3
/model_1_dense_4_biasadd_readvariableop_resource2
.model_1_dense_5_matmul_readvariableop_resource3
/model_1_dense_5_biasadd_readvariableop_resource
identityИвgaussian_noise2/ReadVariableOpв gaussian_noise2/ReadVariableOp_1вgaussian_noise2/condв"model/dense/BiasAdd/ReadVariableOpв!model/dense/MatMul/ReadVariableOpв$model/dense_1/BiasAdd/ReadVariableOpв#model/dense_1/MatMul/ReadVariableOpв$model/dense_2/BiasAdd/ReadVariableOpв#model/dense_2/MatMul/ReadVariableOpв&model_1/dense_3/BiasAdd/ReadVariableOpв%model_1/dense_3/MatMul/ReadVariableOpв&model_1/dense_4/BiasAdd/ReadVariableOpв%model_1/dense_4/MatMul/ReadVariableOpв&model_1/dense_5/BiasAdd/ReadVariableOpв%model_1/dense_5/MatMul/ReadVariableOp▒
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!model/dense/MatMul/ReadVariableOpЧ
model/dense/MatMulMatMulinputs)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense/MatMul░
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"model/dense/BiasAdd/ReadVariableOp▒
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model/dense/Relu╖
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#model/dense_1/MatMul/ReadVariableOp╡
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense_1/MatMul╢
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp╣
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense_1/BiasAddВ
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model/dense_1/Relu╖
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_2/MatMul/ReadVariableOp╖
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_2/MatMul╢
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp╣
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_2/BiasAddm
model/lambda/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model/lambda/pow/yЪ
model/lambda/powPowmodel/dense_2/BiasAdd:output:0model/lambda/pow/y:output:0*
T0*'
_output_shapes
:         2
model/lambda/powm
model/lambda/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lambda/add/yТ
model/lambda/addAddV2model/lambda/pow:z:0model/lambda/add/y:output:0*
T0*'
_output_shapes
:         2
model/lambda/addМ
#model/lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/lambda/Mean/reduction_indicesм
model/lambda/MeanMeanmodel/lambda/add:z:0,model/lambda/Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
model/lambda/Means
model/lambda/SqrtSqrtmodel/lambda/Mean:output:0*
T0*
_output_shapes

:2
model/lambda/Sqrtа
model/lambda/truedivRealDivmodel/dense_2/BiasAdd:output:0model/lambda/Sqrt:y:0*
T0*'
_output_shapes
:         2
model/lambda/truedivд
gaussian_noise2/ReadVariableOpReadVariableOp'gaussian_noise2_readvariableop_resource*
_output_shapes
:*
dtype02 
gaussian_noise2/ReadVariableOpФ
#gaussian_noise2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#gaussian_noise2/strided_slice/stackШ
%gaussian_noise2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%gaussian_noise2/strided_slice/stack_1Ш
%gaussian_noise2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%gaussian_noise2/strided_slice/stack_2╩
gaussian_noise2/strided_sliceStridedSlice&gaussian_noise2/ReadVariableOp:value:0,gaussian_noise2/strided_slice/stack:output:0.gaussian_noise2/strided_slice/stack_1:output:0.gaussian_noise2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gaussian_noise2/strided_sliceи
 gaussian_noise2/ReadVariableOp_1ReadVariableOp'gaussian_noise2_readvariableop_resource*
_output_shapes
:*
dtype02"
 gaussian_noise2/ReadVariableOp_1Ш
%gaussian_noise2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%gaussian_noise2/strided_slice_1/stackЬ
'gaussian_noise2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'gaussian_noise2/strided_slice_1/stack_1Ь
'gaussian_noise2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'gaussian_noise2/strided_slice_1/stack_2╓
gaussian_noise2/strided_slice_1StridedSlice(gaussian_noise2/ReadVariableOp_1:value:0.gaussian_noise2/strided_slice_1/stack:output:00gaussian_noise2/strided_slice_1/stack_1:output:00gaussian_noise2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
gaussian_noise2/strided_slice_1╩
gaussian_noise2/EqualEqual&gaussian_noise2/strided_slice:output:0(gaussian_noise2/strided_slice_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gaussian_noise2/EqualЯ
gaussian_noise2/condIfgaussian_noise2/Equal:z:0'gaussian_noise2_readvariableop_resourcemodel/lambda/truediv:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
:*#
_read_only_resource_inputs
*4
else_branch%R#
!gaussian_noise2_cond_false_271702*
output_shapes
:*3
then_branch$R"
 gaussian_noise2_cond_true_2717012
gaussian_noise2/condМ
gaussian_noise2/cond/IdentityIdentitygaussian_noise2/cond:output:0*
T0*
_output_shapes
:2
gaussian_noise2/cond/Identityv
gaussian_noise2/ShapeShapemodel/lambda/truediv:z:0*
T0*
_output_shapes
:2
gaussian_noise2/ShapeН
"gaussian_noise2/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"gaussian_noise2/random_normal/meanС
$gaussian_noise2/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2&
$gaussian_noise2/random_normal/stddev¤
2gaussian_noise2/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2╘їQ24
2gaussian_noise2/random_normal/RandomStandardNormalы
!gaussian_noise2/random_normal/mulMul;gaussian_noise2/random_normal/RandomStandardNormal:output:0-gaussian_noise2/random_normal/stddev:output:0*
T0*'
_output_shapes
:         2#
!gaussian_noise2/random_normal/mul╦
gaussian_noise2/random_normalAdd%gaussian_noise2/random_normal/mul:z:0+gaussian_noise2/random_normal/mean:output:0*
T0*'
_output_shapes
:         2
gaussian_noise2/random_normalЯ
gaussian_noise2/mulMul&gaussian_noise2/cond/Identity:output:0!gaussian_noise2/random_normal:z:0*
T0*
_output_shapes
:2
gaussian_noise2/mulШ
gaussian_noise2/addAddV2model/lambda/truediv:z:0gaussian_noise2/mul:z:0*
T0*'
_output_shapes
:         2
gaussian_noise2/add╜
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%model_1/dense_3/MatMul/ReadVariableOp┤
model_1/dense_3/MatMulMatMulgaussian_noise2/add:z:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_1/dense_3/MatMul╝
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&model_1/dense_3/BiasAdd/ReadVariableOp┴
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_1/dense_3/BiasAddИ
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model_1/dense_3/Relu╜
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02'
%model_1/dense_4/MatMul/ReadVariableOp┐
model_1/dense_4/MatMulMatMul"model_1/dense_3/Relu:activations:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_1/dense_4/MatMul╝
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&model_1/dense_4/BiasAdd/ReadVariableOp┴
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_1/dense_4/BiasAddИ
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model_1/dense_4/Relu╜
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%model_1/dense_5/MatMul/ReadVariableOp┐
model_1/dense_5/MatMulMatMul"model_1/dense_4/Relu:activations:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_1/dense_5/MatMul╝
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_5/BiasAdd/ReadVariableOp┴
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_1/dense_5/BiasAddе
IdentityIdentity model_1/dense_5/BiasAdd:output:0^gaussian_noise2/ReadVariableOp!^gaussian_noise2/ReadVariableOp_1^gaussian_noise2/cond#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::2@
gaussian_noise2/ReadVariableOpgaussian_noise2/ReadVariableOp2D
 gaussian_noise2/ReadVariableOp_1 gaussian_noise2/ReadVariableOp_12,
gaussian_noise2/condgaussian_noise2/cond2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╒-
■
!gaussian_noise2_cond_false_2718160
,gaussian_noise2_cond_readvariableop_resource3
/gaussian_noise2_cond_shape_model_lambda_truediv!
gaussian_noise2_cond_identityИв#gaussian_noise2/cond/ReadVariableOpв%gaussian_noise2/cond/ReadVariableOp_1Ч
gaussian_noise2/cond/ShapeShape/gaussian_noise2_cond_shape_model_lambda_truediv*
T0*
_output_shapes
:2
gaussian_noise2/cond/ShapeЮ
(gaussian_noise2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gaussian_noise2/cond/strided_slice/stackв
*gaussian_noise2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*gaussian_noise2/cond/strided_slice/stack_1в
*gaussian_noise2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gaussian_noise2/cond/strided_slice/stack_2р
"gaussian_noise2/cond/strided_sliceStridedSlice#gaussian_noise2/cond/Shape:output:01gaussian_noise2/cond/strided_slice/stack:output:03gaussian_noise2/cond/strided_slice/stack_1:output:03gaussian_noise2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"gaussian_noise2/cond/strided_sliceЫ
gaussian_noise2/cond/Shape_1Shape/gaussian_noise2_cond_shape_model_lambda_truediv*
T0*
_output_shapes
:2
gaussian_noise2/cond/Shape_1│
#gaussian_noise2/cond/ReadVariableOpReadVariableOp,gaussian_noise2_cond_readvariableop_resource*
_output_shapes
:*
dtype02%
#gaussian_noise2/cond/ReadVariableOpв
*gaussian_noise2/cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*gaussian_noise2/cond/strided_slice_1/stackж
,gaussian_noise2/cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,gaussian_noise2/cond/strided_slice_1/stack_1ж
,gaussian_noise2/cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gaussian_noise2/cond/strided_slice_1/stack_2Є
$gaussian_noise2/cond/strided_slice_1StridedSlice+gaussian_noise2/cond/ReadVariableOp:value:03gaussian_noise2/cond/strided_slice_1/stack:output:05gaussian_noise2/cond/strided_slice_1/stack_1:output:05gaussian_noise2/cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$gaussian_noise2/cond/strided_slice_1Л
gaussian_noise2/cond/LogLog-gaussian_noise2/cond/strided_slice_1:output:0*
T0*
_output_shapes
: 2
gaussian_noise2/cond/Log╖
%gaussian_noise2/cond/ReadVariableOp_1ReadVariableOp,gaussian_noise2_cond_readvariableop_resource*
_output_shapes
:*
dtype02'
%gaussian_noise2/cond/ReadVariableOp_1в
*gaussian_noise2/cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*gaussian_noise2/cond/strided_slice_2/stackж
,gaussian_noise2/cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,gaussian_noise2/cond/strided_slice_2/stack_1ж
,gaussian_noise2/cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,gaussian_noise2/cond/strided_slice_2/stack_2Ї
$gaussian_noise2/cond/strided_slice_2StridedSlice-gaussian_noise2/cond/ReadVariableOp_1:value:03gaussian_noise2/cond/strided_slice_2/stack:output:05gaussian_noise2/cond/strided_slice_2/stack_1:output:05gaussian_noise2/cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$gaussian_noise2/cond/strided_slice_2П
gaussian_noise2/cond/Log_1Log-gaussian_noise2/cond/strided_slice_2:output:0*
T0*
_output_shapes
: 2
gaussian_noise2/cond/Log_1Ь
+gaussian_noise2/cond/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2-
+gaussian_noise2/cond/random_uniform/shape/1я
)gaussian_noise2/cond/random_uniform/shapePack+gaussian_noise2/cond/strided_slice:output:04gaussian_noise2/cond/random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2+
)gaussian_noise2/cond/random_uniform/shapeЙ
1gaussian_noise2/cond/random_uniform/RandomUniformRandomUniform2gaussian_noise2/cond/random_uniform/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2И╚Ш23
1gaussian_noise2/cond/random_uniform/RandomUniform╕
'gaussian_noise2/cond/random_uniform/subSubgaussian_noise2/cond/Log_1:y:0gaussian_noise2/cond/Log:y:0*
T0*
_output_shapes
: 2)
'gaussian_noise2/cond/random_uniform/subЇ
'gaussian_noise2/cond/random_uniform/mulMul:gaussian_noise2/cond/random_uniform/RandomUniform:output:0+gaussian_noise2/cond/random_uniform/sub:z:0*
T0*'
_output_shapes
:         2)
'gaussian_noise2/cond/random_uniform/mul╬
#gaussian_noise2/cond/random_uniformAdd+gaussian_noise2/cond/random_uniform/mul:z:0gaussian_noise2/cond/Log:y:0*
T0*'
_output_shapes
:         2%
#gaussian_noise2/cond/random_uniformЦ
gaussian_noise2/cond/ExpExp'gaussian_noise2/cond/random_uniform:z:0*
T0*'
_output_shapes
:         2
gaussian_noise2/cond/Expш
gaussian_noise2/cond/IdentityIdentitygaussian_noise2/cond/Exp:y:0$^gaussian_noise2/cond/ReadVariableOp&^gaussian_noise2/cond/ReadVariableOp_1*
T0*'
_output_shapes
:         2
gaussian_noise2/cond/Identity"G
gaussian_noise2_cond_identity&gaussian_noise2/cond/Identity:output:0**
_input_shapes
::         2J
#gaussian_noise2/cond/ReadVariableOp#gaussian_noise2/cond/ReadVariableOp2N
%gaussian_noise2/cond/ReadVariableOp_1%gaussian_noise2/cond/ReadVariableOp_1:-)
'
_output_shapes
:         
Ы
░
C__inference_model_1_layer_call_and_return_conditional_losses_272167

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpе
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOpЛ
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_3/Reluе
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOpЯ
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_4/Reluе
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddп
IdentityIdentitydense_5/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:@ <

_output_shapes
:
 
_user_specified_nameinputs
Ы
░
C__inference_model_1_layer_call_and_return_conditional_losses_271382

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpе
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOpЛ
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_3/Reluе
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOpЯ
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_4/Reluе
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddп
IdentityIdentitydense_5/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:@ <

_output_shapes
:
 
_user_specified_nameinputs
┼	
е
(__inference_model_2_layer_call_fn_271617
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         */
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2715882
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_3
└
╣
(__inference_model_1_layer_call_fn_272201

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2714062
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*/
_input_shapes
:::::::22
StatefulPartitionedCallStatefulPartitionedCall:@ <

_output_shapes
:
 
_user_specified_nameinputs
к
░
C__inference_model_1_layer_call_and_return_conditional_losses_272249

inputs*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource
identityИвdense_3/BiasAdd/ReadVariableOpвdense_3/MatMul/ReadVariableOpвdense_4/BiasAdd/ReadVariableOpвdense_4/MatMul/ReadVariableOpвdense_5/BiasAdd/ReadVariableOpвdense_5/MatMul/ReadVariableOpе
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOpЛ
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/MatMulд
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOpб
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_3/Reluе
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOpЯ
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/MatMulд
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOpб
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_4/Reluе
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOpЯ
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/MatMulд
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOpб
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_5/BiasAddп
IdentityIdentitydense_5/BiasAdd:output:0^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
э	
▄
C__inference_dense_4_layer_call_and_return_conditional_losses_271095

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
э	
▄
C__inference_dense_3_layer_call_and_return_conditional_losses_271068

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▌
}
(__inference_dense_5_layer_call_fn_272435

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2711212
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┬	
д
(__inference_model_2_layer_call_fn_271917

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityИвStatefulPartitionedCallЗ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         */
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2715242
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▌
}
(__inference_dense_2_layer_call_fn_272342

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2709042
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╣j
ж

C__inference_model_2_layer_call_and_return_conditional_losses_271886

inputs.
*model_dense_matmul_readvariableop_resource/
+model_dense_biasadd_readvariableop_resource0
,model_dense_1_matmul_readvariableop_resource1
-model_dense_1_biasadd_readvariableop_resource0
,model_dense_2_matmul_readvariableop_resource1
-model_dense_2_biasadd_readvariableop_resource+
'gaussian_noise2_readvariableop_resource2
.model_1_dense_3_matmul_readvariableop_resource3
/model_1_dense_3_biasadd_readvariableop_resource2
.model_1_dense_4_matmul_readvariableop_resource3
/model_1_dense_4_biasadd_readvariableop_resource2
.model_1_dense_5_matmul_readvariableop_resource3
/model_1_dense_5_biasadd_readvariableop_resource
identityИвgaussian_noise2/ReadVariableOpв gaussian_noise2/ReadVariableOp_1вgaussian_noise2/condв"model/dense/BiasAdd/ReadVariableOpв!model/dense/MatMul/ReadVariableOpв$model/dense_1/BiasAdd/ReadVariableOpв#model/dense_1/MatMul/ReadVariableOpв$model/dense_2/BiasAdd/ReadVariableOpв#model/dense_2/MatMul/ReadVariableOpв&model_1/dense_3/BiasAdd/ReadVariableOpв%model_1/dense_3/MatMul/ReadVariableOpв&model_1/dense_4/BiasAdd/ReadVariableOpв%model_1/dense_4/MatMul/ReadVariableOpв&model_1/dense_5/BiasAdd/ReadVariableOpв%model_1/dense_5/MatMul/ReadVariableOp▒
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!model/dense/MatMul/ReadVariableOpЧ
model/dense/MatMulMatMulinputs)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense/MatMul░
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"model/dense/BiasAdd/ReadVariableOp▒
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model/dense/Relu╖
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#model/dense_1/MatMul/ReadVariableOp╡
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense_1/MatMul╢
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp╣
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model/dense_1/BiasAddВ
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model/dense_1/Relu╖
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_2/MatMul/ReadVariableOp╖
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_2/MatMul╢
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp╣
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model/dense_2/BiasAddm
model/lambda/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model/lambda/pow/yЪ
model/lambda/powPowmodel/dense_2/BiasAdd:output:0model/lambda/pow/y:output:0*
T0*'
_output_shapes
:         2
model/lambda/powm
model/lambda/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lambda/add/yТ
model/lambda/addAddV2model/lambda/pow:z:0model/lambda/add/y:output:0*
T0*'
_output_shapes
:         2
model/lambda/addМ
#model/lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/lambda/Mean/reduction_indicesм
model/lambda/MeanMeanmodel/lambda/add:z:0,model/lambda/Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
model/lambda/Means
model/lambda/SqrtSqrtmodel/lambda/Mean:output:0*
T0*
_output_shapes

:2
model/lambda/Sqrtа
model/lambda/truedivRealDivmodel/dense_2/BiasAdd:output:0model/lambda/Sqrt:y:0*
T0*'
_output_shapes
:         2
model/lambda/truedivд
gaussian_noise2/ReadVariableOpReadVariableOp'gaussian_noise2_readvariableop_resource*
_output_shapes
:*
dtype02 
gaussian_noise2/ReadVariableOpФ
#gaussian_noise2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#gaussian_noise2/strided_slice/stackШ
%gaussian_noise2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%gaussian_noise2/strided_slice/stack_1Ш
%gaussian_noise2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%gaussian_noise2/strided_slice/stack_2╩
gaussian_noise2/strided_sliceStridedSlice&gaussian_noise2/ReadVariableOp:value:0,gaussian_noise2/strided_slice/stack:output:0.gaussian_noise2/strided_slice/stack_1:output:0.gaussian_noise2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
gaussian_noise2/strided_sliceи
 gaussian_noise2/ReadVariableOp_1ReadVariableOp'gaussian_noise2_readvariableop_resource*
_output_shapes
:*
dtype02"
 gaussian_noise2/ReadVariableOp_1Ш
%gaussian_noise2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2'
%gaussian_noise2/strided_slice_1/stackЬ
'gaussian_noise2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'gaussian_noise2/strided_slice_1/stack_1Ь
'gaussian_noise2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'gaussian_noise2/strided_slice_1/stack_2╓
gaussian_noise2/strided_slice_1StridedSlice(gaussian_noise2/ReadVariableOp_1:value:0.gaussian_noise2/strided_slice_1/stack:output:00gaussian_noise2/strided_slice_1/stack_1:output:00gaussian_noise2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
gaussian_noise2/strided_slice_1╩
gaussian_noise2/EqualEqual&gaussian_noise2/strided_slice:output:0(gaussian_noise2/strided_slice_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
gaussian_noise2/EqualЯ
gaussian_noise2/condIfgaussian_noise2/Equal:z:0'gaussian_noise2_readvariableop_resourcemodel/lambda/truediv:z:0*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
:*#
_read_only_resource_inputs
*4
else_branch%R#
!gaussian_noise2_cond_false_271816*
output_shapes
:*3
then_branch$R"
 gaussian_noise2_cond_true_2718152
gaussian_noise2/condМ
gaussian_noise2/cond/IdentityIdentitygaussian_noise2/cond:output:0*
T0*
_output_shapes
:2
gaussian_noise2/cond/Identityv
gaussian_noise2/ShapeShapemodel/lambda/truediv:z:0*
T0*
_output_shapes
:2
gaussian_noise2/ShapeН
"gaussian_noise2/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2$
"gaussian_noise2/random_normal/meanС
$gaussian_noise2/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2&
$gaussian_noise2/random_normal/stddev■
2gaussian_noise2/random_normal/RandomStandardNormalRandomStandardNormalgaussian_noise2/Shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2╔яН24
2gaussian_noise2/random_normal/RandomStandardNormalы
!gaussian_noise2/random_normal/mulMul;gaussian_noise2/random_normal/RandomStandardNormal:output:0-gaussian_noise2/random_normal/stddev:output:0*
T0*'
_output_shapes
:         2#
!gaussian_noise2/random_normal/mul╦
gaussian_noise2/random_normalAdd%gaussian_noise2/random_normal/mul:z:0+gaussian_noise2/random_normal/mean:output:0*
T0*'
_output_shapes
:         2
gaussian_noise2/random_normalЯ
gaussian_noise2/mulMul&gaussian_noise2/cond/Identity:output:0!gaussian_noise2/random_normal:z:0*
T0*
_output_shapes
:2
gaussian_noise2/mulШ
gaussian_noise2/addAddV2model/lambda/truediv:z:0gaussian_noise2/mul:z:0*
T0*'
_output_shapes
:         2
gaussian_noise2/add╜
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%model_1/dense_3/MatMul/ReadVariableOp┤
model_1/dense_3/MatMulMatMulgaussian_noise2/add:z:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_1/dense_3/MatMul╝
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&model_1/dense_3/BiasAdd/ReadVariableOp┴
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_1/dense_3/BiasAddИ
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model_1/dense_3/Relu╜
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02'
%model_1/dense_4/MatMul/ReadVariableOp┐
model_1/dense_4/MatMulMatMul"model_1/dense_3/Relu:activations:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_1/dense_4/MatMul╝
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&model_1/dense_4/BiasAdd/ReadVariableOp┴
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
model_1/dense_4/BiasAddИ
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:          2
model_1/dense_4/Relu╜
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%model_1/dense_5/MatMul/ReadVariableOp┐
model_1/dense_5/MatMulMatMul"model_1/dense_4/Relu:activations:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_1/dense_5/MatMul╝
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_5/BiasAdd/ReadVariableOp┴
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
model_1/dense_5/BiasAddе
IdentityIdentity model_1/dense_5/BiasAdd:output:0^gaussian_noise2/ReadVariableOp!^gaussian_noise2/ReadVariableOp_1^gaussian_noise2/cond#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::2@
gaussian_noise2/ReadVariableOpgaussian_noise2/ReadVariableOp2D
 gaussian_noise2/ReadVariableOp_1 gaussian_noise2/ReadVariableOp_12,
gaussian_noise2/condgaussian_noise2/cond2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2F
!model/dense/MatMul/ReadVariableOp!model/dense/MatMul/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2J
#model/dense_2/MatMul/ReadVariableOp#model/dense_2/MatMul/ReadVariableOp2P
&model_1/dense_3/BiasAdd/ReadVariableOp&model_1/dense_3/BiasAdd/ReadVariableOp2N
%model_1/dense_3/MatMul/ReadVariableOp%model_1/dense_3/MatMul/ReadVariableOp2P
&model_1/dense_4/BiasAdd/ReadVariableOp&model_1/dense_4/BiasAdd/ReadVariableOp2N
%model_1/dense_4/MatMul/ReadVariableOp%model_1/dense_4/MatMul/ReadVariableOp2P
&model_1/dense_5/BiasAdd/ReadVariableOp&model_1/dense_5/BiasAdd/ReadVariableOp2N
%model_1/dense_5/MatMul/ReadVariableOp%model_1/dense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
▌
}
(__inference_dense_4_layer_call_fn_272416

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2710952
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
╗#
ж
A__inference_model_layer_call_and_return_conditional_losses_272012

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOpЕ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:          2

dense/Reluе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_1/MatMul/ReadVariableOpЭ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_1/Reluе
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOpЯ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/BiasAdda
lambda/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda/pow/yВ

lambda/powPowdense_2/BiasAdd:output:0lambda/pow/y:output:0*
T0*'
_output_shapes
:         2

lambda/powa
lambda/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda/add/yz

lambda/addAddV2lambda/pow:z:0lambda/add/y:output:0*
T0*'
_output_shapes
:         2

lambda/addА
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
lambda/Mean/reduction_indicesФ
lambda/MeanMeanlambda/add:z:0&lambda/Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
lambda/Meana
lambda/SqrtSqrtlambda/Mean:output:0*
T0*
_output_shapes

:2
lambda/SqrtИ
lambda/truedivRealDivdense_2/BiasAdd:output:0lambda/Sqrt:y:0*
T0*'
_output_shapes
:         2
lambda/truedivе
IdentityIdentitylambda/truediv:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
·
╕
A__inference_model_layer_call_and_return_conditional_losses_271038

inputs
dense_271021
dense_271023
dense_1_271026
dense_1_271028
dense_2_271031
dense_2_271033
identityИвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallИ
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_271021dense_271023*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2708512
dense/StatefulPartitionedCall▓
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_271026dense_1_271028*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2708782!
dense_1/StatefulPartitionedCall┤
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_271031dense_2_271033*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2709042!
dense_2/StatefulPartitionedCallє
lambda/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2709442
lambda/PartitionedCall╫
IdentityIdentitylambda/PartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
щ
╧
C__inference_model_2_layer_call_and_return_conditional_losses_271455
input_3
model_271268
model_271270
model_271272
model_271274
model_271276
model_271278
gaussian_noise2_271356
model_1_271441
model_1_271443
model_1_271445
model_1_271447
model_1_271449
model_1_271451
identityИв'gaussian_noise2/StatefulPartitionedCallвmodel/StatefulPartitionedCallвmodel_1/StatefulPartitionedCall╔
model/StatefulPartitionedCallStatefulPartitionedCallinput_3model_271268model_271270model_271272model_271274model_271276model_271278*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2710012
model/StatefulPartitionedCall▒
'gaussian_noise2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0gaussian_noise2_271356*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_2713472)
'gaussian_noise2/StatefulPartitionedCallД
model_1/StatefulPartitionedCallStatefulPartitionedCall0gaussian_noise2/StatefulPartitionedCall:output:0model_1_271441model_1_271443model_1_271445model_1_271447model_1_271449model_1_271451*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2713822!
model_1/StatefulPartitionedCallш
IdentityIdentity(model_1/StatefulPartitionedCall:output:0(^gaussian_noise2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::2R
'gaussian_noise2/StatefulPartitionedCall'gaussian_noise2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_3
с
║
(__inference_model_1_layer_call_fn_271230
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallл
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2712152
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_2
│!
Р
cond_false_271297 
cond_readvariableop_resource
cond_shape_inputs
cond_identityИвcond/ReadVariableOpвcond/ReadVariableOp_1Y

cond/ShapeShapecond_shape_inputs*
T0*
_output_shapes
:2

cond/Shape~
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice/stackВ
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1В
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2А
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice]
cond/Shape_1Shapecond_shape_inputs*
T0*
_output_shapes
:2
cond/Shape_1Г
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOpВ
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stackЖ
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_1/stack_1Ж
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_1/stack_2Т
cond/strided_slice_1StridedSlicecond/ReadVariableOp:value:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_1[
cond/LogLogcond/strided_slice_1:output:0*
T0*
_output_shapes
: 2

cond/LogЗ
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp_1В
cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stackЖ
cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack_1Ж
cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack_2Ф
cond/strided_slice_2StridedSlicecond/ReadVariableOp_1:value:0#cond/strided_slice_2/stack:output:0%cond/strided_slice_2/stack_1:output:0%cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_2_

cond/Log_1Logcond/strided_slice_2:output:0*
T0*
_output_shapes
: 2

cond/Log_1|
cond/random_uniform/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
cond/random_uniform/shape/1п
cond/random_uniform/shapePackcond/strided_slice:output:0$cond/random_uniform/shape/1:output:0*
N*
T0*
_output_shapes
:2
cond/random_uniform/shape┘
!cond/random_uniform/RandomUniformRandomUniform"cond/random_uniform/shape:output:0*
T0*'
_output_shapes
:         *
dtype0*
seed▒ х)*
seed2лЬб2#
!cond/random_uniform/RandomUniformx
cond/random_uniform/subSubcond/Log_1:y:0cond/Log:y:0*
T0*
_output_shapes
: 2
cond/random_uniform/sub┤
cond/random_uniform/mulMul*cond/random_uniform/RandomUniform:output:0cond/random_uniform/sub:z:0*
T0*'
_output_shapes
:         2
cond/random_uniform/mulО
cond/random_uniformAddcond/random_uniform/mul:z:0cond/Log:y:0*
T0*'
_output_shapes
:         2
cond/random_uniformf
cond/ExpExpcond/random_uniform:z:0*
T0*'
_output_shapes
:         2

cond/ExpШ
cond/IdentityIdentitycond/Exp:y:0^cond/ReadVariableOp^cond/ReadVariableOp_1*
T0*'
_output_shapes
:         2
cond/Identity"'
cond_identitycond/Identity:output:0**
_input_shapes
::         2*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:-)
'
_output_shapes
:         
├	
v
cond_true_271296 
cond_readvariableop_resource
cond_placeholder
cond_identityИвcond/ReadVariableOpГ
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp~
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice/stackВ
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1В
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2И
cond/strided_sliceStridedSlicecond/ReadVariableOp:value:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice~
cond/IdentityIdentitycond/strided_slice:output:0^cond/ReadVariableOp*
T0*
_output_shapes
: 2
cond/Identity"'
cond_identitycond/Identity:output:0**
_input_shapes
::         2*
cond/ReadVariableOpcond/ReadVariableOp:-)
'
_output_shapes
:         
┘
{
&__inference_dense_layer_call_fn_272303

inputs
unknown
	unknown_0
identityИвStatefulPartitionedCallЇ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:          *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2708512
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:         ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
С
C
'__inference_lambda_layer_call_fn_272376

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2709442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ж
╞
 gaussian_noise2_cond_true_2717010
,gaussian_noise2_cond_readvariableop_resource$
 gaussian_noise2_cond_placeholder!
gaussian_noise2_cond_identityИв#gaussian_noise2/cond/ReadVariableOp│
#gaussian_noise2/cond/ReadVariableOpReadVariableOp,gaussian_noise2_cond_readvariableop_resource*
_output_shapes
:*
dtype02%
#gaussian_noise2/cond/ReadVariableOpЮ
(gaussian_noise2/cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(gaussian_noise2/cond/strided_slice/stackв
*gaussian_noise2/cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*gaussian_noise2/cond/strided_slice/stack_1в
*gaussian_noise2/cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*gaussian_noise2/cond/strided_slice/stack_2ш
"gaussian_noise2/cond/strided_sliceStridedSlice+gaussian_noise2/cond/ReadVariableOp:value:01gaussian_noise2/cond/strided_slice/stack:output:03gaussian_noise2/cond/strided_slice/stack_1:output:03gaussian_noise2/cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"gaussian_noise2/cond/strided_slice╛
gaussian_noise2/cond/IdentityIdentity+gaussian_noise2/cond/strided_slice:output:0$^gaussian_noise2/cond/ReadVariableOp*
T0*
_output_shapes
: 2
gaussian_noise2/cond/Identity"G
gaussian_noise2_cond_identity&gaussian_noise2/cond/Identity:output:0**
_input_shapes
::         2J
#gaussian_noise2/cond/ReadVariableOp#gaussian_noise2/cond/ReadVariableOp:-)
'
_output_shapes
:         
С	
▄
C__inference_dense_2_layer_call_and_return_conditional_losses_270904

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
э	
▄
C__inference_dense_4_layer_call_and_return_conditional_losses_272407

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:          2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:          2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
С	
▄
C__inference_dense_5_layer_call_and_return_conditional_losses_271121

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2	
BiasAddХ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*.
_input_shapes
:          ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:          
 
_user_specified_nameinputs
┼	
е
(__inference_model_2_layer_call_fn_271553
input_3
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11
identityИвStatefulPartitionedCallИ
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         */
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8В *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2715242
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*Z
_input_shapesI
G:         :::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:         
!
_user_specified_name	input_3
┌
╖
&__inference_model_layer_call_fn_272029

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИвStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2710012
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╗#
ж
A__inference_model_layer_call_and_return_conditional_losses_271980

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource
identityИвdense/BiasAdd/ReadVariableOpвdense/MatMul/ReadVariableOpвdense_1/BiasAdd/ReadVariableOpвdense_1/MatMul/ReadVariableOpвdense_2/BiasAdd/ReadVariableOpвdense_2/MatMul/ReadVariableOpЯ
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOpЕ
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense/MatMulЮ
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOpЩ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:          2

dense/Reluе
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_1/MatMul/ReadVariableOpЭ
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/MatMulд
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOpб
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:          2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:          2
dense_1/Reluе
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOpЯ
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/MatMulд
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOpб
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         2
dense_2/BiasAdda
lambda/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda/pow/yВ

lambda/powPowdense_2/BiasAdd:output:0lambda/pow/y:output:0*
T0*'
_output_shapes
:         2

lambda/powa
lambda/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
lambda/add/yz

lambda/addAddV2lambda/pow:z:0lambda/add/y:output:0*
T0*'
_output_shapes
:         2

lambda/addА
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
lambda/Mean/reduction_indicesФ
lambda/MeanMeanlambda/add:z:0&lambda/Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
lambda/Meana
lambda/SqrtSqrtlambda/Mean:output:0*
T0*
_output_shapes

:2
lambda/SqrtИ
lambda/truedivRealDivdense_2/BiasAdd:output:0lambda/Sqrt:y:0*
T0*'
_output_shapes
:         2
lambda/truedivе
IdentityIdentitylambda/truediv:z:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:         ::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
о
v
0__inference_gaussian_noise2_layer_call_fn_272119

inputs
unknown
identityИвStatefulPartitionedCallт
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
:*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_2713472
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
:2

Identity"
identityIdentity:output:0**
_input_shapes
:         :22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
НY
у
__inference__traced_save_272590
file_prefix5
1savev2_gaussian_noise2_stddev_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardж
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename╪
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*ъ
valueрB▌-B6layer_with_weights-1/stddev/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesт
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesн
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_gaussian_noise2_stddev_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *;
dtypes1
/2-	2
SaveV2║
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesб
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*═
_input_shapes╗
╕: :: : : : : : : :  : : :: : :  : : :: : : : :  : : :: : :  : : :: : :  : : :: : :  : : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$	 

_output_shapes

:  : 


_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: : 

_output_shapes
::$ 

_output_shapes

: : 

_output_shapes
: :$ 

_output_shapes

:  : 

_output_shapes
: :$ 

_output_shapes

: :  

_output_shapes
::$! 

_output_shapes

: : "

_output_shapes
: :$# 

_output_shapes

:  : $

_output_shapes
: :$% 

_output_shapes

: : &

_output_shapes
::$' 

_output_shapes

: : (

_output_shapes
: :$) 

_output_shapes

:  : *

_output_shapes
: :$+ 

_output_shapes

: : ,

_output_shapes
::-

_output_shapes
: 
▀
^
B__inference_lambda_layer_call_and_return_conditional_losses_272354

inputs
identityS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y[
powPowinputspow/y:output:0*
T0*'
_output_shapes
:         2
powS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/y^
addAddV2pow:z:0add/y:output:0*
T0*'
_output_shapes
:         2
addr
Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
Mean/reduction_indicesx
MeanMeanadd:z:0Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
MeanL
SqrtSqrtMean:output:0*
T0*
_output_shapes

:2
Sqrta
truedivRealDivinputsSqrt:y:0*
T0*'
_output_shapes
:         2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:         2

Identity"
identityIdentity:output:0*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╓╖
╝
"__inference__traced_restore_272732
file_prefix+
'assignvariableop_gaussian_noise2_stddev 
assignvariableop_1_adam_iter"
assignvariableop_2_adam_beta_1"
assignvariableop_3_adam_beta_2!
assignvariableop_4_adam_decay)
%assignvariableop_5_adam_learning_rate#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias%
!assignvariableop_8_dense_1_kernel#
assignvariableop_9_dense_1_bias&
"assignvariableop_10_dense_2_kernel$
 assignvariableop_11_dense_2_bias&
"assignvariableop_12_dense_3_kernel$
 assignvariableop_13_dense_3_bias&
"assignvariableop_14_dense_4_kernel$
 assignvariableop_15_dense_4_bias&
"assignvariableop_16_dense_5_kernel$
 assignvariableop_17_dense_5_bias
assignvariableop_18_total
assignvariableop_19_count+
'assignvariableop_20_adam_dense_kernel_m)
%assignvariableop_21_adam_dense_bias_m-
)assignvariableop_22_adam_dense_1_kernel_m+
'assignvariableop_23_adam_dense_1_bias_m-
)assignvariableop_24_adam_dense_2_kernel_m+
'assignvariableop_25_adam_dense_2_bias_m-
)assignvariableop_26_adam_dense_3_kernel_m+
'assignvariableop_27_adam_dense_3_bias_m-
)assignvariableop_28_adam_dense_4_kernel_m+
'assignvariableop_29_adam_dense_4_bias_m-
)assignvariableop_30_adam_dense_5_kernel_m+
'assignvariableop_31_adam_dense_5_bias_m+
'assignvariableop_32_adam_dense_kernel_v)
%assignvariableop_33_adam_dense_bias_v-
)assignvariableop_34_adam_dense_1_kernel_v+
'assignvariableop_35_adam_dense_1_bias_v-
)assignvariableop_36_adam_dense_2_kernel_v+
'assignvariableop_37_adam_dense_2_bias_v-
)assignvariableop_38_adam_dense_3_kernel_v+
'assignvariableop_39_adam_dense_3_bias_v-
)assignvariableop_40_adam_dense_4_kernel_v+
'assignvariableop_41_adam_dense_4_bias_v-
)assignvariableop_42_adam_dense_5_kernel_v+
'assignvariableop_43_adam_dense_5_bias_v
identity_45ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9▐
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*ъ
valueрB▌-B6layer_with_weights-1/stddev/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesш
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesП
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*╩
_output_shapes╖
┤:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identityж
AssignVariableOpAssignVariableOp'assignvariableop_gaussian_noise2_stddevIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_1б
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_iterIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2г
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3г
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_2Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4в
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_decayIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5к
AssignVariableOp_5AssignVariableOp%assignvariableop_5_adam_learning_rateIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6д
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7в
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8ж
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9д
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10к
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11и
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12к
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13и
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14к
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15и
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16к
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17и
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_5_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18б
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19б
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20п
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21н
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_dense_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22▒
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_1_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23п
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_1_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24▒
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_2_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25п
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_2_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26▒
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_3_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27п
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_3_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28▒
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_4_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29п
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_4_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30▒
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_5_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31п
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_5_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32п
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33н
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_dense_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34▒
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_1_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35п
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_1_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36▒
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_2_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37п
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_2_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38▒
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_3_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39п
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_3_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40▒
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_4_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41п
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_4_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42▒
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_5_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43п
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_dense_5_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_439
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpж
Identity_44Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_44Щ
Identity_45IdentityIdentity_44:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_45"#
identity_45Identity_45:output:0*╟
_input_shapes╡
▓: ::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"▒L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*к
serving_defaultЦ
;
input_30
serving_default_input_3:0         ;
model_10
StatefulPartitionedCall:0         tensorflow/serving/predict:╪г
оX
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
trainable_variables
regularization_losses
	variables
		keras_api


signatures
в__call__
г_default_save_signature
+д&call_and_return_all_conditional_losses"▐U
_tf_keras_network┬U{"class_name": "Functional", "name": "model_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAEwAAAHMOAAAAdABqAXwAiABkAY0CUwApAk6pAdoEYXhpcykC\n2gJtZtoPbm9ybWFsaXplX2lucHV0qQHaAXipAdoGYXhub3JtqQD6SC9ob21lL2JlY2svQXJiZWl0\nL1NlYWZpbGUvUHJvbW90aW9uL1B5dGhvbi9NTF9SZWNlaXZlci9Db21fRXN0aW1hdGlvbi5wedoI\nPGxhbWJkYT6TAwAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [0]}]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["lambda", 0, 0]]}, "name": "model", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "GaussianNoise2", "config": {"name": "gaussian_noise2", "trainable": true, "dtype": "float32", "stddev": [0.15848931924611132, 0.5011872336272722]}, "name": "gaussian_noise2", "inbound_nodes": [[["model", 1, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_5", 0, 0]]}, "name": "model_1", "inbound_nodes": [[["gaussian_noise2", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["model_1", 1, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAEwAAAHMOAAAAdABqAXwAiABkAY0CUwApAk6pAdoEYXhpcykC\n2gJtZtoPbm9ybWFsaXplX2lucHV0qQHaAXipAdoGYXhub3JtqQD6SC9ob21lL2JlY2svQXJiZWl0\nL1NlYWZpbGUvUHJvbW90aW9uL1B5dGhvbi9NTF9SZWNlaXZlci9Db21fRXN0aW1hdGlvbi5wedoI\nPGxhbWJkYT6TAwAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [0]}]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["lambda", 0, 0]]}, "name": "model", "inbound_nodes": [[["input_3", 0, 0, {}]]]}, {"class_name": "GaussianNoise2", "config": {"name": "gaussian_noise2", "trainable": true, "dtype": "float32", "stddev": [0.15848931924611132, 0.5011872336272722]}, "name": "gaussian_noise2", "inbound_nodes": [[["model", 1, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_5", 0, 0]]}, "name": "model_1", "inbound_nodes": [[["gaussian_noise2", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0]], "output_layers": [["model_1", 1, 0]]}}, "training_config": {"loss": "mean_squared_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
щ"ц
_tf_keras_input_layer╞{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}
Ь-
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
trainable_variables
regularization_losses
	variables
	keras_api
е__call__
+ж&call_and_return_all_conditional_losses"№*
_tf_keras_networkр*{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAEwAAAHMOAAAAdABqAXwAiABkAY0CUwApAk6pAdoEYXhpcykC\n2gJtZtoPbm9ybWFsaXplX2lucHV0qQHaAXipAdoGYXhub3JtqQD6SC9ob21lL2JlY2svQXJiZWl0\nL1NlYWZpbGUvUHJvbW90aW9uL1B5dGhvbi9NTF9SZWNlaXZlci9Db21fRXN0aW1hdGlvbi5wedoI\nPGxhbWJkYT6TAwAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [0]}]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["lambda", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Lambda", "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAEwAAAHMOAAAAdABqAXwAiABkAY0CUwApAk6pAdoEYXhpcykC\n2gJtZtoPbm9ybWFsaXplX2lucHV0qQHaAXipAdoGYXhub3JtqQD6SC9ob21lL2JlY2svQXJiZWl0\nL1NlYWZpbGUvUHJvbW90aW9uL1B5dGhvbi9NTF9SZWNlaXZlci9Db21fRXN0aW1hdGlvbi5wedoI\nPGxhbWJkYT6TAwAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [0]}]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}, "name": "lambda", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["lambda", 0, 0]]}}}
╒

stddev
trainable_variables
regularization_losses
	variables
	keras_api
з__call__
+и&call_and_return_all_conditional_losses"╕
_tf_keras_layerЮ{"class_name": "GaussianNoise2", "name": "gaussian_noise2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "gaussian_noise2", "trainable": true, "dtype": "float32", "stddev": [0.15848931924611132, 0.5011872336272722]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
я"
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
trainable_variables
regularization_losses
	variables
 	keras_api
й__call__
+к&call_and_return_all_conditional_losses"▄ 
_tf_keras_network└ {"class_name": "Functional", "name": "model_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_5", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 16]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}], "input_layers": [["input_2", 0, 0]], "output_layers": [["dense_5", 0, 0]]}}}
├
!iter

"beta_1

#beta_2
	$decay
%learning_rate&mК'mЛ(mМ)mН*mО+mП,mР-mС.mТ/mУ0mФ1mХ&vЦ'vЧ(vШ)vЩ*vЪ+vЫ,vЬ-vЭ.vЮ/vЯ0vа1vб"
	optimizer
v
&0
'1
(2
)3
*4
+5
,6
-7
.8
/9
010
111"
trackable_list_wrapper
 "
trackable_list_wrapper
~
&0
'1
(2
)3
*4
+5
6
,7
-8
.9
/10
011
112"
trackable_list_wrapper
╬
trainable_variables

2layers
3metrics
regularization_losses
	variables
4layer_regularization_losses
5layer_metrics
6non_trainable_variables
в__call__
г_default_save_signature
+д&call_and_return_all_conditional_losses
'д"call_and_return_conditional_losses"
_generic_user_object
-
лserving_default"
signature_map
щ"ц
_tf_keras_input_layer╞{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
ш

&kernel
'bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
м__call__
+н&call_and_return_all_conditional_losses"┴
_tf_keras_layerз{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
ю

(kernel
)bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
о__call__
+п&call_and_return_all_conditional_losses"╟
_tf_keras_layerн{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
Ї

*kernel
+bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
░__call__
+▒&call_and_return_all_conditional_losses"═
_tf_keras_layer│{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 16, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
о
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
▓__call__
+│&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Lambda", "name": "lambda", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "lambda", "trainable": true, "dtype": "float32", "function": {"class_name": "__tuple__", "items": ["4wEAAAAAAAAAAAAAAAEAAAAEAAAAEwAAAHMOAAAAdABqAXwAiABkAY0CUwApAk6pAdoEYXhpcykC\n2gJtZtoPbm9ybWFsaXplX2lucHV0qQHaAXipAdoGYXhub3JtqQD6SC9ob21lL2JlY2svQXJiZWl0\nL1NlYWZpbGUvUHJvbW90aW9uL1B5dGhvbi9NTF9SZWNlaXZlci9Db21fRXN0aW1hdGlvbi5wedoI\nPGxhbWJkYT6TAwAA8wAAAAA=\n", null, {"class_name": "__tuple__", "items": [0]}]}, "function_type": "lambda", "module": "__main__", "output_shape": null, "output_shape_type": "raw", "output_shape_module": null, "arguments": {}}}
J
&0
'1
(2
)3
*4
+5"
trackable_list_wrapper
 "
trackable_list_wrapper
J
&0
'1
(2
)3
*4
+5"
trackable_list_wrapper
░
trainable_variables

Glayers
Hmetrics
regularization_losses
	variables
Ilayer_regularization_losses
Jlayer_metrics
Knon_trainable_variables
е__call__
+ж&call_and_return_all_conditional_losses
'ж"call_and_return_conditional_losses"
_generic_user_object
": 2gaussian_noise2/stddev
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
░
trainable_variables

Llayers
Mmetrics
regularization_losses
	variables
Nlayer_regularization_losses
Olayer_metrics
Pnon_trainable_variables
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
ы"ш
_tf_keras_input_layer╚{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 16]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}
ю

,kernel
-bias
Qtrainable_variables
Rregularization_losses
S	variables
T	keras_api
┤__call__
+╡&call_and_return_all_conditional_losses"╟
_tf_keras_layerн{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16]}}
ю

.kernel
/bias
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
╢__call__
+╖&call_and_return_all_conditional_losses"╟
_tf_keras_layerн{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 32, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
є

0kernel
1bias
Ytrainable_variables
Zregularization_losses
[	variables
\	keras_api
╕__call__
+╣&call_and_return_all_conditional_losses"╠
_tf_keras_layer▓{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32]}}
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
 "
trackable_list_wrapper
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
░
trainable_variables

]layers
^metrics
regularization_losses
	variables
_layer_regularization_losses
`layer_metrics
anon_trainable_variables
й__call__
+к&call_and_return_all_conditional_losses
'к"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
: 2dense/kernel
: 2
dense/bias
 :  2dense_1/kernel
: 2dense_1/bias
 : 2dense_2/kernel
:2dense_2/bias
 : 2dense_3/kernel
: 2dense_3/bias
 :  2dense_4/kernel
: 2dense_4/bias
 : 2dense_5/kernel
:2dense_5/bias
<
0
1
2
3"
trackable_list_wrapper
'
b0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
░
7trainable_variables

clayers
dmetrics
8regularization_losses
9	variables
elayer_regularization_losses
flayer_metrics
gnon_trainable_variables
м__call__
+н&call_and_return_all_conditional_losses
'н"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
░
;trainable_variables

hlayers
imetrics
<regularization_losses
=	variables
jlayer_regularization_losses
klayer_metrics
lnon_trainable_variables
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
░
?trainable_variables

mlayers
nmetrics
@regularization_losses
A	variables
olayer_regularization_losses
player_metrics
qnon_trainable_variables
░__call__
+▒&call_and_return_all_conditional_losses
'▒"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
░
Ctrainable_variables

rlayers
smetrics
Dregularization_losses
E	variables
tlayer_regularization_losses
ulayer_metrics
vnon_trainable_variables
▓__call__
+│&call_and_return_all_conditional_losses
'│"call_and_return_conditional_losses"
_generic_user_object
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
'
0"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
░
Qtrainable_variables

wlayers
xmetrics
Rregularization_losses
S	variables
ylayer_regularization_losses
zlayer_metrics
{non_trainable_variables
┤__call__
+╡&call_and_return_all_conditional_losses
'╡"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
▒
Utrainable_variables

|layers
}metrics
Vregularization_losses
W	variables
~layer_regularization_losses
layer_metrics
Аnon_trainable_variables
╢__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
╡
Ytrainable_variables
Бlayers
Вmetrics
Zregularization_losses
[	variables
 Гlayer_regularization_losses
Дlayer_metrics
Еnon_trainable_variables
╕__call__
+╣&call_and_return_all_conditional_losses
'╣"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
┐

Жtotal

Зcount
И	variables
Й	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
:  (2total
:  (2count
0
Ж0
З1"
trackable_list_wrapper
.
И	variables"
_generic_user_object
#:! 2Adam/dense/kernel/m
: 2Adam/dense/bias/m
%:#  2Adam/dense_1/kernel/m
: 2Adam/dense_1/bias/m
%:# 2Adam/dense_2/kernel/m
:2Adam/dense_2/bias/m
%:# 2Adam/dense_3/kernel/m
: 2Adam/dense_3/bias/m
%:#  2Adam/dense_4/kernel/m
: 2Adam/dense_4/bias/m
%:# 2Adam/dense_5/kernel/m
:2Adam/dense_5/bias/m
#:! 2Adam/dense/kernel/v
: 2Adam/dense/bias/v
%:#  2Adam/dense_1/kernel/v
: 2Adam/dense_1/bias/v
%:# 2Adam/dense_2/kernel/v
:2Adam/dense_2/bias/v
%:# 2Adam/dense_3/kernel/v
: 2Adam/dense_3/bias/v
%:#  2Adam/dense_4/kernel/v
: 2Adam/dense_4/bias/v
%:# 2Adam/dense_5/kernel/v
:2Adam/dense_5/bias/v
ю2ы
(__inference_model_2_layer_call_fn_271948
(__inference_model_2_layer_call_fn_271553
(__inference_model_2_layer_call_fn_271617
(__inference_model_2_layer_call_fn_271917└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
▀2▄
!__inference__wrapped_model_270836╢
Л▓З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *&в#
!К
input_3         
┌2╫
C__inference_model_2_layer_call_and_return_conditional_losses_271886
C__inference_model_2_layer_call_and_return_conditional_losses_271488
C__inference_model_2_layer_call_and_return_conditional_losses_271455
C__inference_model_2_layer_call_and_return_conditional_losses_271772└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ц2у
&__inference_model_layer_call_fn_272029
&__inference_model_layer_call_fn_272046
&__inference_model_layer_call_fn_271053
&__inference_model_layer_call_fn_271016└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
A__inference_model_layer_call_and_return_conditional_losses_270958
A__inference_model_layer_call_and_return_conditional_losses_272012
A__inference_model_layer_call_and_return_conditional_losses_270978
A__inference_model_layer_call_and_return_conditional_losses_271980└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
┌2╫
0__inference_gaussian_noise2_layer_call_fn_272119в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ї2Є
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_272112в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┬2┐
(__inference_model_1_layer_call_fn_271194
(__inference_model_1_layer_call_fn_272184
(__inference_model_1_layer_call_fn_272201
(__inference_model_1_layer_call_fn_272283
(__inference_model_1_layer_call_fn_272266
(__inference_model_1_layer_call_fn_271230└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
ф2с
C__inference_model_1_layer_call_and_return_conditional_losses_272143
C__inference_model_1_layer_call_and_return_conditional_losses_271138
C__inference_model_1_layer_call_and_return_conditional_losses_272167
C__inference_model_1_layer_call_and_return_conditional_losses_272225
C__inference_model_1_layer_call_and_return_conditional_losses_272249
C__inference_model_1_layer_call_and_return_conditional_losses_271157└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╦B╚
$__inference_signature_wrapper_271658input_3"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╨2═
&__inference_dense_layer_call_fn_272303в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ы2ш
A__inference_dense_layer_call_and_return_conditional_losses_272294в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_1_layer_call_fn_272323в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_1_layer_call_and_return_conditional_losses_272314в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_2_layer_call_fn_272342в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_2_layer_call_and_return_conditional_losses_272333в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
Ш2Х
'__inference_lambda_layer_call_fn_272371
'__inference_lambda_layer_call_fn_272376└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╬2╦
B__inference_lambda_layer_call_and_return_conditional_losses_272354
B__inference_lambda_layer_call_and_return_conditional_losses_272366└
╖▓│
FullArgSpec1
args)Ъ&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaultsЪ

 
p 

kwonlyargsЪ 
kwonlydefaultsк 
annotationsк *
 
╥2╧
(__inference_dense_3_layer_call_fn_272396в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_3_layer_call_and_return_conditional_losses_272387в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_4_layer_call_fn_272416в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_4_layer_call_and_return_conditional_losses_272407в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╥2╧
(__inference_dense_5_layer_call_fn_272435в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
э2ъ
C__inference_dense_5_layer_call_and_return_conditional_losses_272426в
Щ▓Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 Щ
!__inference__wrapped_model_270836t&'()*+,-./010в-
&в#
!К
input_3         
к "1к.
,
model_1!К
model_1         г
C__inference_dense_1_layer_call_and_return_conditional_losses_272314\()/в,
%в"
 К
inputs          
к "%в"
К
0          
Ъ {
(__inference_dense_1_layer_call_fn_272323O()/в,
%в"
 К
inputs          
к "К          г
C__inference_dense_2_layer_call_and_return_conditional_losses_272333\*+/в,
%в"
 К
inputs          
к "%в"
К
0         
Ъ {
(__inference_dense_2_layer_call_fn_272342O*+/в,
%в"
 К
inputs          
к "К         г
C__inference_dense_3_layer_call_and_return_conditional_losses_272387\,-/в,
%в"
 К
inputs         
к "%в"
К
0          
Ъ {
(__inference_dense_3_layer_call_fn_272396O,-/в,
%в"
 К
inputs         
к "К          г
C__inference_dense_4_layer_call_and_return_conditional_losses_272407\.//в,
%в"
 К
inputs          
к "%в"
К
0          
Ъ {
(__inference_dense_4_layer_call_fn_272416O.//в,
%в"
 К
inputs          
к "К          г
C__inference_dense_5_layer_call_and_return_conditional_losses_272426\01/в,
%в"
 К
inputs          
к "%в"
К
0         
Ъ {
(__inference_dense_5_layer_call_fn_272435O01/в,
%в"
 К
inputs          
к "К         б
A__inference_dense_layer_call_and_return_conditional_losses_272294\&'/в,
%в"
 К
inputs         
к "%в"
К
0          
Ъ y
&__inference_dense_layer_call_fn_272303O&'/в,
%в"
 К
inputs         
к "К          Ы
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_272112L/в,
%в"
 К
inputs         
к "в
К	
0
Ъ s
0__inference_gaussian_noise2_layer_call_fn_272119?/в,
%в"
 К
inputs         
к "	Кж
B__inference_lambda_layer_call_and_return_conditional_losses_272354`7в4
-в*
 К
inputs         

 
p
к "%в"
К
0         
Ъ ж
B__inference_lambda_layer_call_and_return_conditional_losses_272366`7в4
-в*
 К
inputs         

 
p 
к "%в"
К
0         
Ъ ~
'__inference_lambda_layer_call_fn_272371S7в4
-в*
 К
inputs         

 
p
к "К         ~
'__inference_lambda_layer_call_fn_272376S7в4
-в*
 К
inputs         

 
p 
к "К         ░
C__inference_model_1_layer_call_and_return_conditional_losses_271138i,-./018в5
.в+
!К
input_2         
p

 
к "%в"
К
0         
Ъ ░
C__inference_model_1_layer_call_and_return_conditional_losses_271157i,-./018в5
.в+
!К
input_2         
p 

 
к "%в"
К
0         
Ъ а
C__inference_model_1_layer_call_and_return_conditional_losses_272143Y,-./01(в%
в
К
inputs
p

 
к "%в"
К
0         
Ъ а
C__inference_model_1_layer_call_and_return_conditional_losses_272167Y,-./01(в%
в
К
inputs
p 

 
к "%в"
К
0         
Ъ п
C__inference_model_1_layer_call_and_return_conditional_losses_272225h,-./017в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ п
C__inference_model_1_layer_call_and_return_conditional_losses_272249h,-./017в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ И
(__inference_model_1_layer_call_fn_271194\,-./018в5
.в+
!К
input_2         
p

 
к "К         И
(__inference_model_1_layer_call_fn_271230\,-./018в5
.в+
!К
input_2         
p 

 
к "К         x
(__inference_model_1_layer_call_fn_272184L,-./01(в%
в
К
inputs
p

 
к "К         x
(__inference_model_1_layer_call_fn_272201L,-./01(в%
в
К
inputs
p 

 
к "К         З
(__inference_model_1_layer_call_fn_272266[,-./017в4
-в*
 К
inputs         
p

 
к "К         З
(__inference_model_1_layer_call_fn_272283[,-./017в4
-в*
 К
inputs         
p 

 
к "К         ╖
C__inference_model_2_layer_call_and_return_conditional_losses_271455p&'()*+,-./018в5
.в+
!К
input_3         
p

 
к "%в"
К
0         
Ъ ╖
C__inference_model_2_layer_call_and_return_conditional_losses_271488p&'()*+,-./018в5
.в+
!К
input_3         
p 

 
к "%в"
К
0         
Ъ ╢
C__inference_model_2_layer_call_and_return_conditional_losses_271772o&'()*+,-./017в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ ╢
C__inference_model_2_layer_call_and_return_conditional_losses_271886o&'()*+,-./017в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ П
(__inference_model_2_layer_call_fn_271553c&'()*+,-./018в5
.в+
!К
input_3         
p

 
к "К         П
(__inference_model_2_layer_call_fn_271617c&'()*+,-./018в5
.в+
!К
input_3         
p 

 
к "К         О
(__inference_model_2_layer_call_fn_271917b&'()*+,-./017в4
-в*
 К
inputs         
p

 
к "К         О
(__inference_model_2_layer_call_fn_271948b&'()*+,-./017в4
-в*
 К
inputs         
p 

 
к "К         о
A__inference_model_layer_call_and_return_conditional_losses_270958i&'()*+8в5
.в+
!К
input_1         
p

 
к "%в"
К
0         
Ъ о
A__inference_model_layer_call_and_return_conditional_losses_270978i&'()*+8в5
.в+
!К
input_1         
p 

 
к "%в"
К
0         
Ъ н
A__inference_model_layer_call_and_return_conditional_losses_271980h&'()*+7в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ н
A__inference_model_layer_call_and_return_conditional_losses_272012h&'()*+7в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ Ж
&__inference_model_layer_call_fn_271016\&'()*+8в5
.в+
!К
input_1         
p

 
к "К         Ж
&__inference_model_layer_call_fn_271053\&'()*+8в5
.в+
!К
input_1         
p 

 
к "К         Е
&__inference_model_layer_call_fn_272029[&'()*+7в4
-в*
 К
inputs         
p

 
к "К         Е
&__inference_model_layer_call_fn_272046[&'()*+7в4
-в*
 К
inputs         
p 

 
к "К         з
$__inference_signature_wrapper_271658&'()*+,-./01;в8
в 
1к.
,
input_3!К
input_3         "1к.
,
model_1!К
model_1         