��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
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
incompatible_shape_errorbool(�
.
Identity

input"T
output"T"	
Ttype
�
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
 �
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
�
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
delete_old_dirsbool(�
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
8
Pow
x"T
y"T
z"T"
Ttype:
2
	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
@
ReadVariableOp
resource
value"dtype"
dtypetype�
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
�
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
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
�
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
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.6.22unknown8��
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�
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
�E
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�E
value�EB�E B�E
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
 
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
^

stddev
	variables
trainable_variables
regularization_losses
	keras_api
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
 	keras_api
�
!iter

"beta_1

#beta_2
	$decay
%learning_rate&m�'m�(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�&v�'v�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�
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
�
2metrics
3layer_regularization_losses

4layers
	variables
5non_trainable_variables
6layer_metrics
trainable_variables
regularization_losses
 
 
h

&kernel
'bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
h

(kernel
)bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
h

*kernel
+bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
R
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
*
&0
'1
(2
)3
*4
+5
*
&0
'1
(2
)3
*4
+5
 
�
Gmetrics
Hlayer_regularization_losses

Ilayers
	variables
Jnon_trainable_variables
Klayer_metrics
trainable_variables
regularization_losses
b`
VARIABLE_VALUEgaussian_noise2/stddev6layer_with_weights-1/stddev/.ATTRIBUTES/VARIABLE_VALUE

0
 
 
�
Lmetrics
Mlayer_regularization_losses

Nlayers
	variables
Onon_trainable_variables
Player_metrics
trainable_variables
regularization_losses
 
h

,kernel
-bias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
h

.kernel
/bias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
h

0kernel
1bias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
*
,0
-1
.2
/3
04
15
*
,0
-1
.2
/3
04
15
 
�
]metrics
^layer_regularization_losses

_layers
	variables
`non_trainable_variables
alayer_metrics
trainable_variables
regularization_losses
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
HF
VARIABLE_VALUEdense/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUE
dense/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_3/kernel&variables/7/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEdense_3/bias&variables/8/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUEdense_4/kernel&variables/9/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_4/bias'variables/10/.ATTRIBUTES/VARIABLE_VALUE
KI
VARIABLE_VALUEdense_5/kernel'variables/11/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense_5/bias'variables/12/.ATTRIBUTES/VARIABLE_VALUE

b0
 

0
1
2
3

0
 

&0
'1

&0
'1
 
�
cmetrics
dlayer_regularization_losses

elayers
7	variables
fnon_trainable_variables
glayer_metrics
8trainable_variables
9regularization_losses

(0
)1

(0
)1
 
�
hmetrics
ilayer_regularization_losses

jlayers
;	variables
knon_trainable_variables
llayer_metrics
<trainable_variables
=regularization_losses

*0
+1

*0
+1
 
�
mmetrics
nlayer_regularization_losses

olayers
?	variables
pnon_trainable_variables
qlayer_metrics
@trainable_variables
Aregularization_losses
 
 
 
�
rmetrics
slayer_regularization_losses

tlayers
C	variables
unon_trainable_variables
vlayer_metrics
Dtrainable_variables
Eregularization_losses
 
 
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

0
 

,0
-1

,0
-1
 
�
wmetrics
xlayer_regularization_losses

ylayers
Q	variables
znon_trainable_variables
{layer_metrics
Rtrainable_variables
Sregularization_losses

.0
/1

.0
/1
 
�
|metrics
}layer_regularization_losses

~layers
U	variables
non_trainable_variables
�layer_metrics
Vtrainable_variables
Wregularization_losses

00
11

00
11
 
�
�metrics
 �layer_regularization_losses
�layers
Y	variables
�non_trainable_variables
�layer_metrics
Ztrainable_variables
[regularization_losses
 
 

0
1
2
3
 
 
8

�total

�count
�	variables
�	keras_api
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
�0
�1

�	variables
ki
VARIABLE_VALUEAdam/dense/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_2/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_2/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_3/kernel/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_3/bias/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_4/kernel/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_4/bias/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_5/kernel/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_5/bias/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUEAdam/dense/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_2/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_2/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_3/kernel/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUEAdam/dense_3/bias/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUEAdam/dense_4/kernel/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_4/bias/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUEAdam/dense_5/kernel/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense_5/bias/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_3Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_3dense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasgaussian_noise2/stddevdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference_signature_wrapper_259382
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
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
GPU2*0J 8� *(
f#R!
__inference__traced_save_260359
�
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
GPU2*0J 8� *+
f&R$
"__inference__traced_restore_260501��
�	
�
&__inference_model_layer_call_fn_259571

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2585272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_258686
input_1
dense_258669: 
dense_258671:  
dense_1_258674:  
dense_1_258676:  
dense_2_258679: 
dense_2_258681:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_258669dense_258671*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2584732
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_258674dense_1_258676*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2584902!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_258679dense_2_258681*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2585062!
dense_2/StatefulPartitionedCall�
lambda/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2585242
lambda/PartitionedCallz
IdentityIdentitylambda/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�M
�

C__inference_model_2_layer_call_and_return_conditional_losses_259554

inputs<
*model_dense_matmul_readvariableop_resource: 9
+model_dense_biasadd_readvariableop_resource: >
,model_dense_1_matmul_readvariableop_resource:  ;
-model_dense_1_biasadd_readvariableop_resource: >
,model_dense_2_matmul_readvariableop_resource: ;
-model_dense_2_biasadd_readvariableop_resource:$
gaussian_noise2_259530:@
.model_1_dense_3_matmul_readvariableop_resource: =
/model_1_dense_3_biasadd_readvariableop_resource: @
.model_1_dense_4_matmul_readvariableop_resource:  =
/model_1_dense_4_biasadd_readvariableop_resource: @
.model_1_dense_5_matmul_readvariableop_resource: =
/model_1_dense_5_biasadd_readvariableop_resource:
identity��'gaussian_noise2/StatefulPartitionedCall�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�&model_1/dense_3/BiasAdd/ReadVariableOp�%model_1/dense_3/MatMul/ReadVariableOp�&model_1/dense_4/BiasAdd/ReadVariableOp�%model_1/dense_4/MatMul/ReadVariableOp�&model_1/dense_5/BiasAdd/ReadVariableOp�%model_1/dense_5/MatMul/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!model/dense/MatMul/ReadVariableOp�
model/dense/MatMulMatMulinputs)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model/dense/MatMul�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"model/dense/BiasAdd/ReadVariableOp�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model/dense/Relu�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#model/dense_1/MatMul/ReadVariableOp�
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model/dense_1/MatMul�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model/dense_1/BiasAdd�
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model/dense_1/Relu�
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_2/MatMul/ReadVariableOp�
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_2/MatMul�
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_2/BiasAddm
model/lambda/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model/lambda/pow/y�
model/lambda/powPowmodel/dense_2/BiasAdd:output:0model/lambda/pow/y:output:0*
T0*'
_output_shapes
:���������2
model/lambda/powm
model/lambda/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lambda/add/y�
model/lambda/addAddV2model/lambda/pow:z:0model/lambda/add/y:output:0*
T0*'
_output_shapes
:���������2
model/lambda/add�
#model/lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/lambda/Mean/reduction_indices�
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
model/lambda/Sqrt�
model/lambda/truedivRealDivmodel/dense_2/BiasAdd:output:0model/lambda/Sqrt:y:0*
T0*'
_output_shapes
:���������2
model/lambda/truediv�
'gaussian_noise2/StatefulPartitionedCallStatefulPartitionedCallmodel/lambda/truediv:z:0gaussian_noise2_259530*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� * 
fR
__inference_call_2584302)
'gaussian_noise2/StatefulPartitionedCall�
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%model_1/dense_3/MatMul/ReadVariableOp�
model_1/dense_3/MatMulMatMul0gaussian_noise2/StatefulPartitionedCall:output:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_3/MatMul�
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&model_1/dense_3/BiasAdd/ReadVariableOp�
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_3/BiasAdd�
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_3/Relu�
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02'
%model_1/dense_4/MatMul/ReadVariableOp�
model_1/dense_4/MatMulMatMul"model_1/dense_3/Relu:activations:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_4/MatMul�
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&model_1/dense_4/BiasAdd/ReadVariableOp�
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_4/BiasAdd�
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_4/Relu�
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%model_1/dense_5/MatMul/ReadVariableOp�
model_1/dense_5/MatMulMatMul"model_1/dense_4/Relu:activations:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_1/dense_5/MatMul�
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_5/BiasAdd/ReadVariableOp�
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_1/dense_5/BiasAdd{
IdentityIdentity model_1/dense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp(^gaussian_noise2/StatefulPartitionedCall#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 2R
'gaussian_noise2/StatefulPartitionedCall'gaussian_noise2/StatefulPartitionedCall2H
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
:���������
 
_user_specified_nameinputs
�	
�
&__inference_model_layer_call_fn_258666
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2586342
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_258764

inputs 
dense_3_258725: 
dense_3_258727:  
dense_4_258742:  
dense_4_258744:  
dense_5_258758: 
dense_5_258760:
identity��dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_258725dense_3_258727*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2587242!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_258742dense_4_258744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2587412!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_258758dense_5_258760*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2587572!
dense_5/StatefulPartitionedCall�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�/
�
cond_false_258369*
cond_readvariableop_resource:
cond_shape_inputs
cond_identity��cond/ReadVariableOp�cond/ReadVariableOp_1Y

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
cond/strided_slice/stack�
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1�
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2�
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice�
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stack�
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stack_1�
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_1/stack_2�
cond/strided_slice_1StridedSlicecond/strided_slice:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
new_axis_mask2
cond/strided_slice_1]
cond/Shape_1Shapecond_shape_inputs*
T0*
_output_shapes
:2
cond/Shape_1�
cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack�
cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_2/stack_1�
cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack_2�
cond/strided_slice_2StridedSlicecond/Shape_1:output:0#cond/strided_slice_2/stack:output:0%cond/strided_slice_2/stack_1:output:0%cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
cond/strided_slice_2f
cond/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/Shape_2d
cond/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cond/ones/Constt
	cond/onesFillcond/Shape_2:output:0cond/ones/Const:output:0*
T0*
_output_shapes
:2
	cond/onesf
cond/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
cond/concat/axis�
cond/concatConcatV2cond/strided_slice_1:output:0cond/ones:output:0cond/concat/axis:output:0*
N*
T0*
_output_shapes
:2
cond/concat�
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp�
cond/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_3/stack�
cond/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_3/stack_1�
cond/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_3/stack_2�
cond/strided_slice_3StridedSlicecond/ReadVariableOp:value:0#cond/strided_slice_3/stack:output:0%cond/strided_slice_3/stack_1:output:0%cond/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_3[
cond/LogLogcond/strided_slice_3:output:0*
T0*
_output_shapes
: 2

cond/Log�
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp_1�
cond/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack�
cond/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack_1�
cond/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack_2�
cond/strided_slice_4StridedSlicecond/ReadVariableOp_1:value:0#cond/strided_slice_4/stack:output:0%cond/strided_slice_4/stack_1:output:0%cond/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_4_

cond/Log_1Logcond/strided_slice_4:output:0*
T0*
_output_shapes
: 2

cond/Log_1�
!cond/random_uniform/RandomUniformRandomUniformcond/concat:output:0*
T0*0
_output_shapes
:������������������*
dtype0*
seed���)*
seed2�Ͷ2#
!cond/random_uniform/RandomUniformx
cond/random_uniform/subSubcond/Log_1:y:0cond/Log:y:0*
T0*
_output_shapes
: 2
cond/random_uniform/sub�
cond/random_uniform/mulMul*cond/random_uniform/RandomUniform:output:0cond/random_uniform/sub:z:0*
T0*0
_output_shapes
:������������������2
cond/random_uniform/mul�
cond/random_uniformAddV2cond/random_uniform/mul:z:0cond/Log:y:0*
T0*'
_output_shapes
:���������2
cond/random_uniformf
cond/ExpExpcond/random_uniform:z:0*
T0*'
_output_shapes
:���������2

cond/Expv
cond/IdentityIdentitycond/Exp:y:0
^cond/NoOp*
T0*'
_output_shapes
:���������2
cond/Identity�
	cond/NoOpNoOp^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
	cond/NoOp"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
: :���������2*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:-)
'
_output_shapes
:���������
�
�
(__inference_dense_4_layer_call_fn_260174

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2587412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
$__inference_signature_wrapper_259382
input_3
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6: 
	unknown_7: 
	unknown_8:  
	unknown_9: 

unknown_10: 

unknown_11:
identity��StatefulPartitionedCall�
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
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__wrapped_model_2584552
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_3
�

�
cond_true_258952*
cond_readvariableop_resource:
cond_placeholder
cond_identity��cond/ReadVariableOp�
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
cond/strided_slice/stack�
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1�
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2�
cond/strided_sliceStridedSlicecond/ReadVariableOp:value:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slicet
cond/IdentityIdentitycond/strided_slice:output:0
^cond/NoOp*
T0*
_output_shapes
: 2
cond/Identityn
	cond/NoOpNoOp^cond/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
	cond/NoOp"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
: :���������2*
cond/ReadVariableOpcond/ReadVariableOp:-)
'
_output_shapes
:���������
�
�
__inference_call_259813

inputs%
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�condt
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
strided_slice/stack_2�
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
EqualEqualstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equal�
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
cond_false_259752*
output_shapes
:*#
then_branchR
cond_true_2597512
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
 *  �?2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed���)*
seed2���2$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������2
random_normal/mul�
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������2
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
addS
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
:2

Identityy
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^cond*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_2_layer_call_and_return_conditional_losses_259056

inputs
model_258924: 
model_258926: 
model_258928:  
model_258930: 
model_258932: 
model_258934:$
gaussian_noise2_259015: 
model_1_259042: 
model_1_259044:  
model_1_259046:  
model_1_259048:  
model_1_259050: 
model_1_259052:
identity��'gaussian_noise2/StatefulPartitionedCall�model/StatefulPartitionedCall�model_1/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_258924model_258926model_258928model_258930model_258932model_258934*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2585272
model/StatefulPartitionedCall�
'gaussian_noise2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0gaussian_noise2_259015*
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
GPU2*0J 8� *T
fORM
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_2590142)
'gaussian_noise2/StatefulPartitionedCall�
model_1/StatefulPartitionedCallStatefulPartitionedCall0gaussian_noise2/StatefulPartitionedCall:output:0model_1_259042model_1_259044model_1_259046model_1_259048model_1_259050model_1_259052*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2590412!
model_1/StatefulPartitionedCall�
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp(^gaussian_noise2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 2R
'gaussian_noise2/StatefulPartitionedCall'gaussian_noise2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_258634

inputs
dense_258617: 
dense_258619:  
dense_1_258622:  
dense_1_258624:  
dense_2_258627: 
dense_2_258629:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_258617dense_258619*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2584732
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_258622dense_1_258624*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2584902!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_258627dense_2_258629*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2585062!
dense_2/StatefulPartitionedCall�
lambda/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2585622
lambda/PartitionedCallz
IdentityIdentitylambda/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
^
B__inference_lambda_layer_call_and_return_conditional_losses_258524

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
:���������2
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
:���������2
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
:���������2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_258724

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
&__inference_model_layer_call_fn_259588

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2586342
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_5_layer_call_and_return_conditional_losses_260204

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�

�
C__inference_dense_5_layer_call_and_return_conditional_losses_258757

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�$
�
A__inference_model_layer_call_and_return_conditional_losses_259652

inputs6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:  5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_1/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/BiasAdda
lambda/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda/pow/y�

lambda/powPowdense_2/BiasAdd:output:0lambda/pow/y:output:0*
T0*'
_output_shapes
:���������2

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
:���������2

lambda/add�
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
lambda/Mean/reduction_indices�
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
lambda/Sqrt�
lambda/truedivRealDivdense_2/BiasAdd:output:0lambda/Sqrt:y:0*
T0*'
_output_shapes
:���������2
lambda/truedivm
IdentityIdentitylambda/truediv:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�Y
�
!__inference__wrapped_model_258455
input_3D
2model_2_model_dense_matmul_readvariableop_resource: A
3model_2_model_dense_biasadd_readvariableop_resource: F
4model_2_model_dense_1_matmul_readvariableop_resource:  C
5model_2_model_dense_1_biasadd_readvariableop_resource: F
4model_2_model_dense_2_matmul_readvariableop_resource: C
5model_2_model_dense_2_biasadd_readvariableop_resource:,
model_2_gaussian_noise2_258431:H
6model_2_model_1_dense_3_matmul_readvariableop_resource: E
7model_2_model_1_dense_3_biasadd_readvariableop_resource: H
6model_2_model_1_dense_4_matmul_readvariableop_resource:  E
7model_2_model_1_dense_4_biasadd_readvariableop_resource: H
6model_2_model_1_dense_5_matmul_readvariableop_resource: E
7model_2_model_1_dense_5_biasadd_readvariableop_resource:
identity��/model_2/gaussian_noise2/StatefulPartitionedCall�*model_2/model/dense/BiasAdd/ReadVariableOp�)model_2/model/dense/MatMul/ReadVariableOp�,model_2/model/dense_1/BiasAdd/ReadVariableOp�+model_2/model/dense_1/MatMul/ReadVariableOp�,model_2/model/dense_2/BiasAdd/ReadVariableOp�+model_2/model/dense_2/MatMul/ReadVariableOp�.model_2/model_1/dense_3/BiasAdd/ReadVariableOp�-model_2/model_1/dense_3/MatMul/ReadVariableOp�.model_2/model_1/dense_4/BiasAdd/ReadVariableOp�-model_2/model_1/dense_4/MatMul/ReadVariableOp�.model_2/model_1/dense_5/BiasAdd/ReadVariableOp�-model_2/model_1/dense_5/MatMul/ReadVariableOp�
)model_2/model/dense/MatMul/ReadVariableOpReadVariableOp2model_2_model_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02+
)model_2/model/dense/MatMul/ReadVariableOp�
model_2/model/dense/MatMulMatMulinput_31model_2/model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_2/model/dense/MatMul�
*model_2/model/dense/BiasAdd/ReadVariableOpReadVariableOp3model_2_model_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02,
*model_2/model/dense/BiasAdd/ReadVariableOp�
model_2/model/dense/BiasAddBiasAdd$model_2/model/dense/MatMul:product:02model_2/model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_2/model/dense/BiasAdd�
model_2/model/dense/ReluRelu$model_2/model/dense/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model_2/model/dense/Relu�
+model_2/model/dense_1/MatMul/ReadVariableOpReadVariableOp4model_2_model_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02-
+model_2/model/dense_1/MatMul/ReadVariableOp�
model_2/model/dense_1/MatMulMatMul&model_2/model/dense/Relu:activations:03model_2/model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_2/model/dense_1/MatMul�
,model_2/model/dense_1/BiasAdd/ReadVariableOpReadVariableOp5model_2_model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,model_2/model/dense_1/BiasAdd/ReadVariableOp�
model_2/model/dense_1/BiasAddBiasAdd&model_2/model/dense_1/MatMul:product:04model_2/model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_2/model/dense_1/BiasAdd�
model_2/model/dense_1/ReluRelu&model_2/model/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model_2/model/dense_1/Relu�
+model_2/model/dense_2/MatMul/ReadVariableOpReadVariableOp4model_2_model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02-
+model_2/model/dense_2/MatMul/ReadVariableOp�
model_2/model/dense_2/MatMulMatMul(model_2/model/dense_1/Relu:activations:03model_2/model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_2/model/dense_2/MatMul�
,model_2/model/dense_2/BiasAdd/ReadVariableOpReadVariableOp5model_2_model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,model_2/model/dense_2/BiasAdd/ReadVariableOp�
model_2/model/dense_2/BiasAddBiasAdd&model_2/model/dense_2/MatMul:product:04model_2/model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_2/model/dense_2/BiasAdd}
model_2/model/lambda/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model_2/model/lambda/pow/y�
model_2/model/lambda/powPow&model_2/model/dense_2/BiasAdd:output:0#model_2/model/lambda/pow/y:output:0*
T0*'
_output_shapes
:���������2
model_2/model/lambda/pow}
model_2/model/lambda/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model_2/model/lambda/add/y�
model_2/model/lambda/addAddV2model_2/model/lambda/pow:z:0#model_2/model/lambda/add/y:output:0*
T0*'
_output_shapes
:���������2
model_2/model/lambda/add�
+model_2/model/lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2-
+model_2/model/lambda/Mean/reduction_indices�
model_2/model/lambda/MeanMeanmodel_2/model/lambda/add:z:04model_2/model/lambda/Mean/reduction_indices:output:0*
T0*
_output_shapes

:*
	keep_dims(2
model_2/model/lambda/Mean�
model_2/model/lambda/SqrtSqrt"model_2/model/lambda/Mean:output:0*
T0*
_output_shapes

:2
model_2/model/lambda/Sqrt�
model_2/model/lambda/truedivRealDiv&model_2/model/dense_2/BiasAdd:output:0model_2/model/lambda/Sqrt:y:0*
T0*'
_output_shapes
:���������2
model_2/model/lambda/truediv�
/model_2/gaussian_noise2/StatefulPartitionedCallStatefulPartitionedCall model_2/model/lambda/truediv:z:0model_2_gaussian_noise2_258431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� * 
fR
__inference_call_25843021
/model_2/gaussian_noise2/StatefulPartitionedCall�
-model_2/model_1/dense_3/MatMul/ReadVariableOpReadVariableOp6model_2_model_1_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02/
-model_2/model_1/dense_3/MatMul/ReadVariableOp�
model_2/model_1/dense_3/MatMulMatMul8model_2/gaussian_noise2/StatefulPartitionedCall:output:05model_2/model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
model_2/model_1/dense_3/MatMul�
.model_2/model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp7model_2_model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.model_2/model_1/dense_3/BiasAdd/ReadVariableOp�
model_2/model_1/dense_3/BiasAddBiasAdd(model_2/model_1/dense_3/MatMul:product:06model_2/model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
model_2/model_1/dense_3/BiasAdd�
model_2/model_1/dense_3/ReluRelu(model_2/model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model_2/model_1/dense_3/Relu�
-model_2/model_1/dense_4/MatMul/ReadVariableOpReadVariableOp6model_2_model_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02/
-model_2/model_1/dense_4/MatMul/ReadVariableOp�
model_2/model_1/dense_4/MatMulMatMul*model_2/model_1/dense_3/Relu:activations:05model_2/model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2 
model_2/model_1/dense_4/MatMul�
.model_2/model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp7model_2_model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype020
.model_2/model_1/dense_4/BiasAdd/ReadVariableOp�
model_2/model_1/dense_4/BiasAddBiasAdd(model_2/model_1/dense_4/MatMul:product:06model_2/model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2!
model_2/model_1/dense_4/BiasAdd�
model_2/model_1/dense_4/ReluRelu(model_2/model_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model_2/model_1/dense_4/Relu�
-model_2/model_1/dense_5/MatMul/ReadVariableOpReadVariableOp6model_2_model_1_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02/
-model_2/model_1/dense_5/MatMul/ReadVariableOp�
model_2/model_1/dense_5/MatMulMatMul*model_2/model_1/dense_4/Relu:activations:05model_2/model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2 
model_2/model_1/dense_5/MatMul�
.model_2/model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp7model_2_model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype020
.model_2/model_1/dense_5/BiasAdd/ReadVariableOp�
model_2/model_1/dense_5/BiasAddBiasAdd(model_2/model_1/dense_5/MatMul:product:06model_2/model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2!
model_2/model_1/dense_5/BiasAdd�
IdentityIdentity(model_2/model_1/dense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp0^model_2/gaussian_noise2/StatefulPartitionedCall+^model_2/model/dense/BiasAdd/ReadVariableOp*^model_2/model/dense/MatMul/ReadVariableOp-^model_2/model/dense_1/BiasAdd/ReadVariableOp,^model_2/model/dense_1/MatMul/ReadVariableOp-^model_2/model/dense_2/BiasAdd/ReadVariableOp,^model_2/model/dense_2/MatMul/ReadVariableOp/^model_2/model_1/dense_3/BiasAdd/ReadVariableOp.^model_2/model_1/dense_3/MatMul/ReadVariableOp/^model_2/model_1/dense_4/BiasAdd/ReadVariableOp.^model_2/model_1/dense_4/MatMul/ReadVariableOp/^model_2/model_1/dense_5/BiasAdd/ReadVariableOp.^model_2/model_1/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 2b
/model_2/gaussian_noise2/StatefulPartitionedCall/model_2/gaussian_noise2/StatefulPartitionedCall2X
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
:���������
!
_user_specified_name	input_3
�	
�
(__inference_model_1_layer_call_fn_259905

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2587642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_model_2_layer_call_fn_259277
input_3
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6: 
	unknown_7: 
	unknown_8:  
	unknown_9: 

unknown_10: 

unknown_11:
identity��StatefulPartitionedCall�
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
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2592172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_3
�
�
(__inference_dense_1_layer_call_fn_260081

inputs
unknown:  
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2584902
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
C__inference_model_2_layer_call_and_return_conditional_losses_259343
input_3
model_259313: 
model_259315: 
model_259317:  
model_259319: 
model_259321: 
model_259323:$
gaussian_noise2_259326: 
model_1_259329: 
model_1_259331:  
model_1_259333:  
model_1_259335:  
model_1_259337: 
model_1_259339:
identity��'gaussian_noise2/StatefulPartitionedCall�model/StatefulPartitionedCall�model_1/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinput_3model_259313model_259315model_259317model_259319model_259321model_259323*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2586342
model/StatefulPartitionedCall�
'gaussian_noise2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0gaussian_noise2_259326*
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
GPU2*0J 8� *T
fORM
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_2590142)
'gaussian_noise2/StatefulPartitionedCall�
model_1/StatefulPartitionedCallStatefulPartitionedCall0gaussian_noise2/StatefulPartitionedCall:output:0model_1_259329model_1_259331model_1_259333model_1_259335model_1_259337model_1_259339*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2591282!
model_1/StatefulPartitionedCall�
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp(^gaussian_noise2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 2R
'gaussian_noise2/StatefulPartitionedCall'gaussian_noise2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_3
�

�
cond_true_259751*
cond_readvariableop_resource:
cond_placeholder
cond_identity��cond/ReadVariableOp�
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
cond/strided_slice/stack�
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1�
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2�
cond/strided_sliceStridedSlicecond/ReadVariableOp:value:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slicet
cond/IdentityIdentitycond/strided_slice:output:0
^cond/NoOp*
T0*
_output_shapes
: 2
cond/Identityn
	cond/NoOpNoOp^cond/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
	cond/NoOp"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
: :���������2*
cond/ReadVariableOpcond/ReadVariableOp:-)
'
_output_shapes
:���������
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_258917
input_2 
dense_3_258901: 
dense_3_258903:  
dense_4_258906:  
dense_4_258908:  
dense_5_258911: 
dense_5_258913:
identity��dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_3_258901dense_3_258903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2587242!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_258906dense_4_258908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2587412!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_258911dense_5_258913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2587572!
dense_5/StatefulPartitionedCall�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
C__inference_model_2_layer_call_and_return_conditional_losses_259310
input_3
model_259280: 
model_259282: 
model_259284:  
model_259286: 
model_259288: 
model_259290:$
gaussian_noise2_259293: 
model_1_259296: 
model_1_259298:  
model_1_259300:  
model_1_259302:  
model_1_259304: 
model_1_259306:
identity��'gaussian_noise2/StatefulPartitionedCall�model/StatefulPartitionedCall�model_1/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinput_3model_259280model_259282model_259284model_259286model_259288model_259290*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2585272
model/StatefulPartitionedCall�
'gaussian_noise2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0gaussian_noise2_259293*
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
GPU2*0J 8� *T
fORM
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_2590142)
'gaussian_noise2/StatefulPartitionedCall�
model_1/StatefulPartitionedCallStatefulPartitionedCall0gaussian_noise2/StatefulPartitionedCall:output:0model_1_259296model_1_259298model_1_259300model_1_259302model_1_259304model_1_259306*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2590412!
model_1/StatefulPartitionedCall�
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp(^gaussian_noise2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 2R
'gaussian_noise2/StatefulPartitionedCall'gaussian_noise2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_3
�
C
'__inference_lambda_layer_call_fn_260116

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2585242
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�/
�
cond_false_259752*
cond_readvariableop_resource:
cond_shape_inputs
cond_identity��cond/ReadVariableOp�cond/ReadVariableOp_1Y

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
cond/strided_slice/stack�
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1�
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2�
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice�
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stack�
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stack_1�
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_1/stack_2�
cond/strided_slice_1StridedSlicecond/strided_slice:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
new_axis_mask2
cond/strided_slice_1]
cond/Shape_1Shapecond_shape_inputs*
T0*
_output_shapes
:2
cond/Shape_1�
cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack�
cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_2/stack_1�
cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack_2�
cond/strided_slice_2StridedSlicecond/Shape_1:output:0#cond/strided_slice_2/stack:output:0%cond/strided_slice_2/stack_1:output:0%cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
cond/strided_slice_2f
cond/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/Shape_2d
cond/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cond/ones/Constt
	cond/onesFillcond/Shape_2:output:0cond/ones/Const:output:0*
T0*
_output_shapes
:2
	cond/onesf
cond/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
cond/concat/axis�
cond/concatConcatV2cond/strided_slice_1:output:0cond/ones:output:0cond/concat/axis:output:0*
N*
T0*
_output_shapes
:2
cond/concat�
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp�
cond/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_3/stack�
cond/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_3/stack_1�
cond/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_3/stack_2�
cond/strided_slice_3StridedSlicecond/ReadVariableOp:value:0#cond/strided_slice_3/stack:output:0%cond/strided_slice_3/stack_1:output:0%cond/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_3[
cond/LogLogcond/strided_slice_3:output:0*
T0*
_output_shapes
: 2

cond/Log�
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp_1�
cond/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack�
cond/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack_1�
cond/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack_2�
cond/strided_slice_4StridedSlicecond/ReadVariableOp_1:value:0#cond/strided_slice_4/stack:output:0%cond/strided_slice_4/stack_1:output:0%cond/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_4_

cond/Log_1Logcond/strided_slice_4:output:0*
T0*
_output_shapes
: 2

cond/Log_1�
!cond/random_uniform/RandomUniformRandomUniformcond/concat:output:0*
T0*0
_output_shapes
:������������������*
dtype0*
seed���)*
seed2���2#
!cond/random_uniform/RandomUniformx
cond/random_uniform/subSubcond/Log_1:y:0cond/Log:y:0*
T0*
_output_shapes
: 2
cond/random_uniform/sub�
cond/random_uniform/mulMul*cond/random_uniform/RandomUniform:output:0cond/random_uniform/sub:z:0*
T0*0
_output_shapes
:������������������2
cond/random_uniform/mul�
cond/random_uniformAddV2cond/random_uniform/mul:z:0cond/Log:y:0*
T0*'
_output_shapes
:���������2
cond/random_uniformf
cond/ExpExpcond/random_uniform:z:0*
T0*'
_output_shapes
:���������2

cond/Expv
cond/IdentityIdentitycond/Exp:y:0
^cond/NoOp*
T0*'
_output_shapes
:���������2
cond/Identity�
	cond/NoOpNoOp^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
	cond/NoOp"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
: :���������2*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:-)
'
_output_shapes
:���������
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_258898
input_2 
dense_3_258882: 
dense_3_258884:  
dense_4_258887:  
dense_4_258889:  
dense_5_258892: 
dense_5_258894:
identity��dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_2dense_3_258882dense_3_258884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2587242!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_258887dense_4_258889*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2587412!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_258892dense_5_258894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2587572!
dense_5/StatefulPartitionedCall�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
(__inference_model_2_layer_call_fn_259085
input_3
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6: 
	unknown_7: 
	unknown_8:  
	unknown_9: 

unknown_10: 

unknown_11:
identity��StatefulPartitionedCall�
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
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2590562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_3
�
�
A__inference_model_layer_call_and_return_conditional_losses_258706
input_1
dense_258689: 
dense_258691:  
dense_1_258694:  
dense_1_258696:  
dense_2_258699: 
dense_2_258701:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_258689dense_258691*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2584732
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_258694dense_1_258696*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2584902!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_258699dense_2_258701*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2585062!
dense_2/StatefulPartitionedCall�
lambda/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2585622
lambda/PartitionedCallz
IdentityIdentitylambda/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
C__inference_model_2_layer_call_and_return_conditional_losses_259217

inputs
model_259187: 
model_259189: 
model_259191:  
model_259193: 
model_259195: 
model_259197:$
gaussian_noise2_259200: 
model_1_259203: 
model_1_259205:  
model_1_259207:  
model_1_259209:  
model_1_259211: 
model_1_259213:
identity��'gaussian_noise2/StatefulPartitionedCall�model/StatefulPartitionedCall�model_1/StatefulPartitionedCall�
model/StatefulPartitionedCallStatefulPartitionedCallinputsmodel_259187model_259189model_259191model_259193model_259195model_259197*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2586342
model/StatefulPartitionedCall�
'gaussian_noise2/StatefulPartitionedCallStatefulPartitionedCall&model/StatefulPartitionedCall:output:0gaussian_noise2_259200*
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
GPU2*0J 8� *T
fORM
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_2590142)
'gaussian_noise2/StatefulPartitionedCall�
model_1/StatefulPartitionedCallStatefulPartitionedCall0gaussian_noise2/StatefulPartitionedCall:output:0model_1_259203model_1_259205model_1_259207model_1_259209model_1_259211model_1_259213*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2591282!
model_1/StatefulPartitionedCall�
IdentityIdentity(model_1/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp(^gaussian_noise2/StatefulPartitionedCall^model/StatefulPartitionedCall ^model_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 2R
'gaussian_noise2/StatefulPartitionedCall'gaussian_noise2/StatefulPartitionedCall2>
model/StatefulPartitionedCallmodel/StatefulPartitionedCall2B
model_1/StatefulPartitionedCallmodel_1/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_4_layer_call_and_return_conditional_losses_260185

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
(__inference_dense_5_layer_call_fn_260194

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2587572
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
(__inference_model_1_layer_call_fn_258779
input_2
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2587642
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�

j
cond_true_259828*
cond_readvariableop_resource:
cond_identity��cond/ReadVariableOp�
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
cond/strided_slice/stack�
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1�
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2�
cond/strided_sliceStridedSlicecond/ReadVariableOp:value:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slicet
cond/IdentityIdentitycond/strided_slice:output:0
^cond/NoOp*
T0*
_output_shapes
: 2
cond/Identityn
	cond/NoOpNoOp^cond/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
	cond/NoOp"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2*
cond/ReadVariableOpcond/ReadVariableOp
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_259041

inputs8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_3/Relu�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_4/Relu�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/BiasAdds
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2@
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
�	
^
B__inference_lambda_layer_call_and_return_conditional_losses_258562

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
:���������2
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
:���������2
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
:���������2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_258506

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
&__inference_model_layer_call_fn_258542
input_1
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_model_layer_call_and_return_conditional_losses_2585272
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�M
�

C__inference_model_2_layer_call_and_return_conditional_losses_259499

inputs<
*model_dense_matmul_readvariableop_resource: 9
+model_dense_biasadd_readvariableop_resource: >
,model_dense_1_matmul_readvariableop_resource:  ;
-model_dense_1_biasadd_readvariableop_resource: >
,model_dense_2_matmul_readvariableop_resource: ;
-model_dense_2_biasadd_readvariableop_resource:$
gaussian_noise2_259475:@
.model_1_dense_3_matmul_readvariableop_resource: =
/model_1_dense_3_biasadd_readvariableop_resource: @
.model_1_dense_4_matmul_readvariableop_resource:  =
/model_1_dense_4_biasadd_readvariableop_resource: @
.model_1_dense_5_matmul_readvariableop_resource: =
/model_1_dense_5_biasadd_readvariableop_resource:
identity��'gaussian_noise2/StatefulPartitionedCall�"model/dense/BiasAdd/ReadVariableOp�!model/dense/MatMul/ReadVariableOp�$model/dense_1/BiasAdd/ReadVariableOp�#model/dense_1/MatMul/ReadVariableOp�$model/dense_2/BiasAdd/ReadVariableOp�#model/dense_2/MatMul/ReadVariableOp�&model_1/dense_3/BiasAdd/ReadVariableOp�%model_1/dense_3/MatMul/ReadVariableOp�&model_1/dense_4/BiasAdd/ReadVariableOp�%model_1/dense_4/MatMul/ReadVariableOp�&model_1/dense_5/BiasAdd/ReadVariableOp�%model_1/dense_5/MatMul/ReadVariableOp�
!model/dense/MatMul/ReadVariableOpReadVariableOp*model_dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02#
!model/dense/MatMul/ReadVariableOp�
model/dense/MatMulMatMulinputs)model/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model/dense/MatMul�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"model/dense/BiasAdd/ReadVariableOp�
model/dense/BiasAddBiasAddmodel/dense/MatMul:product:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model/dense/BiasAdd|
model/dense/ReluRelumodel/dense/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model/dense/Relu�
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02%
#model/dense_1/MatMul/ReadVariableOp�
model/dense_1/MatMulMatMulmodel/dense/Relu:activations:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model/dense_1/MatMul�
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp�
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model/dense_1/BiasAdd�
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model/dense_1/Relu�
#model/dense_2/MatMul/ReadVariableOpReadVariableOp,model_dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02%
#model/dense_2/MatMul/ReadVariableOp�
model/dense_2/MatMulMatMul model/dense_1/Relu:activations:0+model/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_2/MatMul�
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$model/dense_2/BiasAdd/ReadVariableOp�
model/dense_2/BiasAddBiasAddmodel/dense_2/MatMul:product:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model/dense_2/BiasAddm
model/lambda/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
model/lambda/pow/y�
model/lambda/powPowmodel/dense_2/BiasAdd:output:0model/lambda/pow/y:output:0*
T0*'
_output_shapes
:���������2
model/lambda/powm
model/lambda/add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
model/lambda/add/y�
model/lambda/addAddV2model/lambda/pow:z:0model/lambda/add/y:output:0*
T0*'
_output_shapes
:���������2
model/lambda/add�
#model/lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2%
#model/lambda/Mean/reduction_indices�
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
model/lambda/Sqrt�
model/lambda/truedivRealDivmodel/dense_2/BiasAdd:output:0model/lambda/Sqrt:y:0*
T0*'
_output_shapes
:���������2
model/lambda/truediv�
'gaussian_noise2/StatefulPartitionedCallStatefulPartitionedCallmodel/lambda/truediv:z:0gaussian_noise2_259475*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� * 
fR
__inference_call_2584302)
'gaussian_noise2/StatefulPartitionedCall�
%model_1/dense_3/MatMul/ReadVariableOpReadVariableOp.model_1_dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%model_1/dense_3/MatMul/ReadVariableOp�
model_1/dense_3/MatMulMatMul0gaussian_noise2/StatefulPartitionedCall:output:0-model_1/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_3/MatMul�
&model_1/dense_3/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&model_1/dense_3/BiasAdd/ReadVariableOp�
model_1/dense_3/BiasAddBiasAdd model_1/dense_3/MatMul:product:0.model_1/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_3/BiasAdd�
model_1/dense_3/ReluRelu model_1/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_3/Relu�
%model_1/dense_4/MatMul/ReadVariableOpReadVariableOp.model_1_dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02'
%model_1/dense_4/MatMul/ReadVariableOp�
model_1/dense_4/MatMulMatMul"model_1/dense_3/Relu:activations:0-model_1/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_4/MatMul�
&model_1/dense_4/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&model_1/dense_4/BiasAdd/ReadVariableOp�
model_1/dense_4/BiasAddBiasAdd model_1/dense_4/MatMul:product:0.model_1/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_4/BiasAdd�
model_1/dense_4/ReluRelu model_1/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
model_1/dense_4/Relu�
%model_1/dense_5/MatMul/ReadVariableOpReadVariableOp.model_1_dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02'
%model_1/dense_5/MatMul/ReadVariableOp�
model_1/dense_5/MatMulMatMul"model_1/dense_4/Relu:activations:0-model_1/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_1/dense_5/MatMul�
&model_1/dense_5/BiasAdd/ReadVariableOpReadVariableOp/model_1_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&model_1/dense_5/BiasAdd/ReadVariableOp�
model_1/dense_5/BiasAddBiasAdd model_1/dense_5/MatMul:product:0.model_1/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
model_1/dense_5/BiasAdd{
IdentityIdentity model_1/dense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp(^gaussian_noise2/StatefulPartitionedCall#^model/dense/BiasAdd/ReadVariableOp"^model/dense/MatMul/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp$^model/dense_2/MatMul/ReadVariableOp'^model_1/dense_3/BiasAdd/ReadVariableOp&^model_1/dense_3/MatMul/ReadVariableOp'^model_1/dense_4/BiasAdd/ReadVariableOp&^model_1/dense_4/MatMul/ReadVariableOp'^model_1/dense_5/BiasAdd/ReadVariableOp&^model_1/dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 2R
'gaussian_noise2/StatefulPartitionedCall'gaussian_noise2/StatefulPartitionedCall2H
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
:���������
 
_user_specified_nameinputs
�
�
__inference_call_258430

inputs%
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�condt
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
strided_slice/stack_2�
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
EqualEqualstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equal�
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
cond_false_258369*
output_shapes
:*#
then_branchR
cond_true_2583682
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
 *  �?2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed���)*
seed2���2$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������2
random_normal/mul�
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������2
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
addS
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
:2

Identityy
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^cond*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
(__inference_model_1_layer_call_fn_259922

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2588472
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�$
�
A__inference_model_layer_call_and_return_conditional_losses_259620

inputs6
$dense_matmul_readvariableop_resource: 3
%dense_biasadd_readvariableop_resource: 8
&dense_1_matmul_readvariableop_resource:  5
'dense_1_biasadd_readvariableop_resource: 8
&dense_2_matmul_readvariableop_resource: 5
'dense_2_biasadd_readvariableop_resource:
identity��dense/BiasAdd/ReadVariableOp�dense/MatMul/ReadVariableOp�dense_1/BiasAdd/ReadVariableOp�dense_1/MatMul/ReadVariableOp�dense_2/BiasAdd/ReadVariableOp�dense_2/MatMul/ReadVariableOp�
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense/MatMul/ReadVariableOp�
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense/MatMul�
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
dense/BiasAdd/ReadVariableOp�
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense/BiasAddj

dense/ReluReludense/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2

dense/Relu�
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_1/MatMul/ReadVariableOp�
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_1/MatMul�
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_1/BiasAdd/ReadVariableOp�
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_1/Relu�
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_2/MatMul/ReadVariableOp�
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/MatMul�
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp�
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_2/BiasAdda
lambda/pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
lambda/pow/y�

lambda/powPowdense_2/BiasAdd:output:0lambda/pow/y:output:0*
T0*'
_output_shapes
:���������2

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
:���������2

lambda/add�
lambda/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
lambda/Mean/reduction_indices�
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
lambda/Sqrt�
lambda/truedivRealDivdense_2/BiasAdd:output:0lambda/Sqrt:y:0*
T0*'
_output_shapes
:���������2
lambda/truedivm
IdentityIdentitylambda/truediv:z:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�.
�
cond_false_259829*
cond_readvariableop_resource:
cond_identity��cond/ReadVariableOp�cond/ReadVariableOp_1i

cond/ShapeConst*
_output_shapes
:*
dtype0*
valueB"�     2

cond/Shape~
cond/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice/stack�
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1�
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2�
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice�
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stack�
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stack_1�
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_1/stack_2�
cond/strided_slice_1StridedSlicecond/strided_slice:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
new_axis_mask2
cond/strided_slice_1m
cond/Shape_1Const*
_output_shapes
:*
dtype0*
valueB"�     2
cond/Shape_1�
cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack�
cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_2/stack_1�
cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack_2�
cond/strided_slice_2StridedSlicecond/Shape_1:output:0#cond/strided_slice_2/stack:output:0%cond/strided_slice_2/stack_1:output:0%cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
cond/strided_slice_2f
cond/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/Shape_2d
cond/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cond/ones/Constt
	cond/onesFillcond/Shape_2:output:0cond/ones/Const:output:0*
T0*
_output_shapes
:2
	cond/onesf
cond/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
cond/concat/axis�
cond/concatConcatV2cond/strided_slice_1:output:0cond/ones:output:0cond/concat/axis:output:0*
N*
T0*
_output_shapes
:2
cond/concat�
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp�
cond/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_3/stack�
cond/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_3/stack_1�
cond/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_3/stack_2�
cond/strided_slice_3StridedSlicecond/ReadVariableOp:value:0#cond/strided_slice_3/stack:output:0%cond/strided_slice_3/stack_1:output:0%cond/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_3[
cond/LogLogcond/strided_slice_3:output:0*
T0*
_output_shapes
: 2

cond/Log�
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp_1�
cond/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack�
cond/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack_1�
cond/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack_2�
cond/strided_slice_4StridedSlicecond/ReadVariableOp_1:value:0#cond/strided_slice_4/stack:output:0%cond/strided_slice_4/stack_1:output:0%cond/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_4_

cond/Log_1Logcond/strided_slice_4:output:0*
T0*
_output_shapes
: 2

cond/Log_1�
!cond/random_uniform/RandomUniformRandomUniformcond/concat:output:0*
T0*
_output_shapes
:	�*
dtype0*
seed���)*
seed2�Ϸ2#
!cond/random_uniform/RandomUniformx
cond/random_uniform/subSubcond/Log_1:y:0cond/Log:y:0*
T0*
_output_shapes
: 2
cond/random_uniform/sub�
cond/random_uniform/mulMul*cond/random_uniform/RandomUniform:output:0cond/random_uniform/sub:z:0*
T0*
_output_shapes
:	�2
cond/random_uniform/mul�
cond/random_uniformAddV2cond/random_uniform/mul:z:0cond/Log:y:0*
T0*
_output_shapes
:	�2
cond/random_uniform^
cond/ExpExpcond/random_uniform:z:0*
T0*
_output_shapes
:	�2

cond/Expn
cond/IdentityIdentitycond/Exp:y:0
^cond/NoOp*
T0*
_output_shapes
:	�2
cond/Identity�
	cond/NoOpNoOp^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
	cond/NoOp"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1
�
�
(__inference_dense_3_layer_call_fn_260154

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2587242
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_258847

inputs 
dense_3_258831: 
dense_3_258833:  
dense_4_258836:  
dense_4_258838:  
dense_5_258841: 
dense_5_258843:
identity��dense_3/StatefulPartitionedCall�dense_4/StatefulPartitionedCall�dense_5/StatefulPartitionedCall�
dense_3/StatefulPartitionedCallStatefulPartitionedCallinputsdense_3_258831dense_3_258833*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_3_layer_call_and_return_conditional_losses_2587242!
dense_3/StatefulPartitionedCall�
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_258836dense_4_258838*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_4_layer_call_and_return_conditional_losses_2587412!
dense_4/StatefulPartitionedCall�
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_258841dense_5_258843*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_5_layer_call_and_return_conditional_losses_2587572!
dense_5/StatefulPartitionedCall�
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_dense_2_layer_call_fn_260101

inputs
unknown: 
	unknown_0:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2585062
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_259736

inputs%
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�condt
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
strided_slice/stack_2�
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
EqualEqualstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equal�
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
cond_false_259675*
output_shapes
:*#
then_branchR
cond_true_2596742
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
 *  �?2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed���)*
seed2��2$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������2
random_normal/mul�
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������2
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
addS
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
:2

Identityy
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^cond*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_4_layer_call_and_return_conditional_losses_258741

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
�
(__inference_model_1_layer_call_fn_258879
input_2
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2588472
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_2
�
�
&__inference_dense_layer_call_fn_260061

inputs
unknown: 
	unknown_0: 
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2584732
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�/
�
cond_false_259675*
cond_readvariableop_resource:
cond_shape_inputs
cond_identity��cond/ReadVariableOp�cond/ReadVariableOp_1Y

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
cond/strided_slice/stack�
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1�
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2�
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice�
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stack�
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stack_1�
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_1/stack_2�
cond/strided_slice_1StridedSlicecond/strided_slice:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
new_axis_mask2
cond/strided_slice_1]
cond/Shape_1Shapecond_shape_inputs*
T0*
_output_shapes
:2
cond/Shape_1�
cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack�
cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_2/stack_1�
cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack_2�
cond/strided_slice_2StridedSlicecond/Shape_1:output:0#cond/strided_slice_2/stack:output:0%cond/strided_slice_2/stack_1:output:0%cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
cond/strided_slice_2f
cond/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/Shape_2d
cond/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cond/ones/Constt
	cond/onesFillcond/Shape_2:output:0cond/ones/Const:output:0*
T0*
_output_shapes
:2
	cond/onesf
cond/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
cond/concat/axis�
cond/concatConcatV2cond/strided_slice_1:output:0cond/ones:output:0cond/concat/axis:output:0*
N*
T0*
_output_shapes
:2
cond/concat�
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp�
cond/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_3/stack�
cond/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_3/stack_1�
cond/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_3/stack_2�
cond/strided_slice_3StridedSlicecond/ReadVariableOp:value:0#cond/strided_slice_3/stack:output:0%cond/strided_slice_3/stack_1:output:0%cond/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_3[
cond/LogLogcond/strided_slice_3:output:0*
T0*
_output_shapes
: 2

cond/Log�
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp_1�
cond/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack�
cond/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack_1�
cond/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack_2�
cond/strided_slice_4StridedSlicecond/ReadVariableOp_1:value:0#cond/strided_slice_4/stack:output:0%cond/strided_slice_4/stack_1:output:0%cond/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_4_

cond/Log_1Logcond/strided_slice_4:output:0*
T0*
_output_shapes
: 2

cond/Log_1�
!cond/random_uniform/RandomUniformRandomUniformcond/concat:output:0*
T0*0
_output_shapes
:������������������*
dtype0*
seed���)*
seed2���2#
!cond/random_uniform/RandomUniformx
cond/random_uniform/subSubcond/Log_1:y:0cond/Log:y:0*
T0*
_output_shapes
: 2
cond/random_uniform/sub�
cond/random_uniform/mulMul*cond/random_uniform/RandomUniform:output:0cond/random_uniform/sub:z:0*
T0*0
_output_shapes
:������������������2
cond/random_uniform/mul�
cond/random_uniformAddV2cond/random_uniform/mul:z:0cond/Log:y:0*
T0*'
_output_shapes
:���������2
cond/random_uniformf
cond/ExpExpcond/random_uniform:z:0*
T0*'
_output_shapes
:���������2

cond/Expv
cond/IdentityIdentitycond/Exp:y:0
^cond/NoOp*
T0*'
_output_shapes
:���������2
cond/Identity�
	cond/NoOpNoOp^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
	cond/NoOp"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
: :���������2*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:-)
'
_output_shapes
:���������
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_260028

inputs8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_3/Relu�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_4/Relu�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/BiasAdds
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2@
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
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_260052

inputs8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_3/Relu�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_4/Relu�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/BiasAdds
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2@
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
�
�
__inference_call_259888

inputs%
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�condt
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
strided_slice/stack_2�
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
EqualEqualstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equal�
condIf	Equal:z:0readvariableop_resource*
Tcond0
*
Tin
2*
Tout
2*
_lower_using_switch_merge(*
_output_shapes
:*#
_read_only_resource_inputs
*$
else_branchR
cond_false_259829*
output_shapes
:*#
then_branchR
cond_true_2598282
cond\
cond/IdentityIdentitycond:output:0*
T0*
_output_shapes
:2
cond/Identity_
ShapeConst*
_output_shapes
:*
dtype0*
valueB"�     2
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
 *  �?2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*
_output_shapes
:	�*
dtype0*
seed���)*
seed2�ʠ2$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*
_output_shapes
:	�2
random_normal/mul�
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*
_output_shapes
:	�2
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
addS
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
:2

Identityy
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^cond*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime* 
_input_shapes
:	�: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12
condcond:G C

_output_shapes
:	�
 
_user_specified_nameinputs
׸
�
"__inference__traced_restore_260501
file_prefix5
'assignvariableop_gaussian_noise2_stddev:&
assignvariableop_1_adam_iter:	 (
assignvariableop_2_adam_beta_1: (
assignvariableop_3_adam_beta_2: '
assignvariableop_4_adam_decay: /
%assignvariableop_5_adam_learning_rate: 1
assignvariableop_6_dense_kernel: +
assignvariableop_7_dense_bias: 3
!assignvariableop_8_dense_1_kernel:  -
assignvariableop_9_dense_1_bias: 4
"assignvariableop_10_dense_2_kernel: .
 assignvariableop_11_dense_2_bias:4
"assignvariableop_12_dense_3_kernel: .
 assignvariableop_13_dense_3_bias: 4
"assignvariableop_14_dense_4_kernel:  .
 assignvariableop_15_dense_4_bias: 4
"assignvariableop_16_dense_5_kernel: .
 assignvariableop_17_dense_5_bias:#
assignvariableop_18_total: #
assignvariableop_19_count: 9
'assignvariableop_20_adam_dense_kernel_m: 3
%assignvariableop_21_adam_dense_bias_m: ;
)assignvariableop_22_adam_dense_1_kernel_m:  5
'assignvariableop_23_adam_dense_1_bias_m: ;
)assignvariableop_24_adam_dense_2_kernel_m: 5
'assignvariableop_25_adam_dense_2_bias_m:;
)assignvariableop_26_adam_dense_3_kernel_m: 5
'assignvariableop_27_adam_dense_3_bias_m: ;
)assignvariableop_28_adam_dense_4_kernel_m:  5
'assignvariableop_29_adam_dense_4_bias_m: ;
)assignvariableop_30_adam_dense_5_kernel_m: 5
'assignvariableop_31_adam_dense_5_bias_m:9
'assignvariableop_32_adam_dense_kernel_v: 3
%assignvariableop_33_adam_dense_bias_v: ;
)assignvariableop_34_adam_dense_1_kernel_v:  5
'assignvariableop_35_adam_dense_1_bias_v: ;
)assignvariableop_36_adam_dense_2_kernel_v: 5
'assignvariableop_37_adam_dense_2_bias_v:;
)assignvariableop_38_adam_dense_3_kernel_v: 5
'assignvariableop_39_adam_dense_3_bias_v: ;
)assignvariableop_40_adam_dense_4_kernel_v:  5
'assignvariableop_41_adam_dense_4_bias_v: ;
)assignvariableop_42_adam_dense_5_kernel_v: 5
'assignvariableop_43_adam_dense_5_bias_v:
identity_45��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*�
value�B�-B6layer_with_weights-1/stddev/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp'assignvariableop_gaussian_noise2_stddevIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_iterIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_1Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_beta_2Identity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_adam_decayIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp%assignvariableop_5_adam_learning_rateIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_5_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOpassignvariableop_18_totalIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOpassignvariableop_19_countIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp'assignvariableop_20_adam_dense_kernel_mIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp%assignvariableop_21_adam_dense_bias_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_dense_1_kernel_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp'assignvariableop_23_adam_dense_1_bias_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_dense_2_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp'assignvariableop_25_adam_dense_2_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp)assignvariableop_26_adam_dense_3_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_3_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_dense_4_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp'assignvariableop_29_adam_dense_4_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_dense_5_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp'assignvariableop_31_adam_dense_5_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_kernel_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp%assignvariableop_33_adam_dense_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_dense_1_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp'assignvariableop_35_adam_dense_1_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_dense_2_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp'assignvariableop_37_adam_dense_2_bias_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_dense_3_kernel_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp'assignvariableop_39_adam_dense_3_bias_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp)assignvariableop_40_adam_dense_4_kernel_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp'assignvariableop_41_adam_dense_4_bias_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_dense_5_kernel_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp'assignvariableop_43_adam_dense_5_bias_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_439
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_44Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_44f
Identity_45IdentityIdentity_44:output:0^NoOp_1*
T0*
_output_shapes
: 2
Identity_45�
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 2
NoOp_1"#
identity_45Identity_45:output:0*m
_input_shapes\
Z: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
_user_specified_namefile_prefix
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_260004

inputs8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_3/Relu�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_4/Relu�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/BiasAdds
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_259014

inputs%
readvariableop_resource:
identity��ReadVariableOp�ReadVariableOp_1�condt
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
strided_slice/stack_2�
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
strided_slice_1/stack_2�
strided_slice_1StridedSliceReadVariableOp_1:value:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1�
EqualEqualstrided_slice:output:0strided_slice_1:output:0*
T0*
_output_shapes
: *
incompatible_shape_error( 2
Equal�
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
cond_false_258953*
output_shapes
:*#
then_branchR
cond_true_2589522
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
 *  �?2
random_normal/stddev�
"random_normal/RandomStandardNormalRandomStandardNormalShape:output:0*
T0*'
_output_shapes
:���������*
dtype0*
seed���)*
seed2ٮ�2$
"random_normal/RandomStandardNormal�
random_normal/mulMul+random_normal/RandomStandardNormal:output:0random_normal/stddev:output:0*
T0*'
_output_shapes
:���������2
random_normal/mul�
random_normalAddV2random_normal/mul:z:0random_normal/mean:output:0*
T0*'
_output_shapes
:���������2
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
addS
IdentityIdentityadd:z:0^NoOp*
T0*
_output_shapes
:2

Identityy
NoOpNoOp^ReadVariableOp^ReadVariableOp_1^cond*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12
condcond:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
A__inference_model_layer_call_and_return_conditional_losses_258527

inputs
dense_258474: 
dense_258476:  
dense_1_258491:  
dense_1_258493:  
dense_2_258507: 
dense_2_258509:
identity��dense/StatefulPartitionedCall�dense_1/StatefulPartitionedCall�dense_2/StatefulPartitionedCall�
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_258474dense_258476*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_2584732
dense/StatefulPartitionedCall�
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_258491dense_1_258493*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:��������� *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_2584902!
dense_1/StatefulPartitionedCall�
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_258507dense_2_258509*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_2_layer_call_and_return_conditional_losses_2585062!
dense_2/StatefulPartitionedCall�
lambda/PartitionedCallPartitionedCall(dense_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2585242
lambda/PartitionedCallz
IdentityIdentitylambda/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_model_1_layer_call_fn_259956

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2591282
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
:: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:@ <

_output_shapes
:
 
_user_specified_nameinputs
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_259980

inputs8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_3/Relu�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_4/Relu�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/BiasAdds
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2@
dense_3/BiasAdd/ReadVariableOpdense_3/BiasAdd/ReadVariableOp2>
dense_3/MatMul/ReadVariableOpdense_3/MatMul/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp2@
dense_5/BiasAdd/ReadVariableOpdense_5/BiasAdd/ReadVariableOp2>
dense_5/MatMul/ReadVariableOpdense_5/MatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
cond_true_258368*
cond_readvariableop_resource:
cond_placeholder
cond_identity��cond/ReadVariableOp�
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
cond/strided_slice/stack�
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1�
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2�
cond/strided_sliceStridedSlicecond/ReadVariableOp:value:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slicet
cond/IdentityIdentitycond/strided_slice:output:0
^cond/NoOp*
T0*
_output_shapes
: 2
cond/Identityn
	cond/NoOpNoOp^cond/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
	cond/NoOp"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
: :���������2*
cond/ReadVariableOpcond/ReadVariableOp:-)
'
_output_shapes
:���������
�
C
'__inference_lambda_layer_call_fn_260121

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *K
fFRD
B__inference_lambda_layer_call_and_return_conditional_losses_2585622
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_dense_3_layer_call_and_return_conditional_losses_260165

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
A__inference_dense_layer_call_and_return_conditional_losses_258473

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
(__inference_model_2_layer_call_fn_259413

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6: 
	unknown_7: 
	unknown_8:  
	unknown_9: 

unknown_10: 

unknown_11:
identity��StatefulPartitionedCall�
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
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2590562
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
0__inference_gaussian_noise2_layer_call_fn_259659

inputs
unknown:
identity��StatefulPartitionedCall�
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
GPU2*0J 8� *T
fORM
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_2590142
StatefulPartitionedCalll
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*
_output_shapes
:2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������: 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
A__inference_dense_layer_call_and_return_conditional_losses_260072

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
C__inference_dense_2_layer_call_and_return_conditional_losses_260111

inputs0
matmul_readvariableop_resource: -
biasadd_readvariableop_resource:
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

: *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddk
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_258490

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�
�
(__inference_model_2_layer_call_fn_259444

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
	unknown_5:
	unknown_6: 
	unknown_7: 
	unknown_8:  
	unknown_9: 

unknown_10: 

unknown_11:
identity��StatefulPartitionedCall�
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
:���������*/
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_2_layer_call_and_return_conditional_losses_2592172
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:���������: : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

�
cond_true_259674*
cond_readvariableop_resource:
cond_placeholder
cond_identity��cond/ReadVariableOp�
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
cond/strided_slice/stack�
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1�
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2�
cond/strided_sliceStridedSlicecond/ReadVariableOp:value:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slicet
cond/IdentityIdentitycond/strided_slice:output:0
^cond/NoOp*
T0*
_output_shapes
: 2
cond/Identityn
	cond/NoOpNoOp^cond/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
	cond/NoOp"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
: :���������2*
cond/ReadVariableOpcond/ReadVariableOp:-)
'
_output_shapes
:���������
�V
�
__inference__traced_save_260359
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

identity_1��MergeV2Checkpoints�
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
Const_1�
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
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*�
value�B�-B6layer_with_weights-1/stddev/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:-*
dtype0*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:01savev2_gaussian_noise2_stddev_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *;
dtypes1
/2-	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: 2

Identity_1c
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"!

identity_1Identity_1:output:0*�
_input_shapes�
�: :: : : : : : : :  : : :: : :  : : :: : : : :  : : :: : :  : : :: : :  : : :: : :  : : :: 2(
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
�
�
C__inference_dense_1_layer_call_and_return_conditional_losses_260092

inputs0
matmul_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:  *
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
Relum
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:��������� 2

Identity
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:��������� : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:��������� 
 
_user_specified_nameinputs
�	
^
B__inference_lambda_layer_call_and_return_conditional_losses_260133

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
:���������2
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
:���������2
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
:���������2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
^
B__inference_lambda_layer_call_and_return_conditional_losses_260145

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
:���������2
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
:���������2
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
:���������2	
truediv_
IdentityIdentitytruediv:z:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_model_1_layer_call_and_return_conditional_losses_259128

inputs8
&dense_3_matmul_readvariableop_resource: 5
'dense_3_biasadd_readvariableop_resource: 8
&dense_4_matmul_readvariableop_resource:  5
'dense_4_biasadd_readvariableop_resource: 8
&dense_5_matmul_readvariableop_resource: 5
'dense_5_biasadd_readvariableop_resource:
identity��dense_3/BiasAdd/ReadVariableOp�dense_3/MatMul/ReadVariableOp�dense_4/BiasAdd/ReadVariableOp�dense_4/MatMul/ReadVariableOp�dense_5/BiasAdd/ReadVariableOp�dense_5/MatMul/ReadVariableOp�
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_3/MatMul/ReadVariableOp�
dense_3/MatMulMatMulinputs%dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/MatMul�
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_3/BiasAdd/ReadVariableOp�
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_3/BiasAddp
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_3/Relu�
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource*
_output_shapes

:  *
dtype02
dense_4/MatMul/ReadVariableOp�
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/MatMul�
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02 
dense_4/BiasAdd/ReadVariableOp�
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:��������� 2
dense_4/BiasAddp
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*'
_output_shapes
:��������� 2
dense_4/Relu�
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource*
_output_shapes

: *
dtype02
dense_5/MatMul/ReadVariableOp�
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/MatMul�
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp�
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_5/BiasAdds
IdentityIdentitydense_5/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identity�
NoOpNoOp^dense_3/BiasAdd/ReadVariableOp^dense_3/MatMul/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp^dense_5/BiasAdd/ReadVariableOp^dense_5/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : : : 2@
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
�
�
(__inference_model_1_layer_call_fn_259939

inputs
unknown: 
	unknown_0: 
	unknown_1:  
	unknown_2: 
	unknown_3: 
	unknown_4:
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_1_layer_call_and_return_conditional_losses_2590412
StatefulPartitionedCall{
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������2

Identityh
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 2
NoOp"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*#
_input_shapes
:: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:@ <

_output_shapes
:
 
_user_specified_nameinputs
�/
�
cond_false_258953*
cond_readvariableop_resource:
cond_shape_inputs
cond_identity��cond/ReadVariableOp�cond/ReadVariableOp_1Y

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
cond/strided_slice/stack�
cond/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_1�
cond/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice/stack_2�
cond/strided_sliceStridedSlicecond/Shape:output:0!cond/strided_slice/stack:output:0#cond/strided_slice/stack_1:output:0#cond/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice�
cond/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stack�
cond/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_1/stack_1�
cond/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_1/stack_2�
cond/strided_slice_1StridedSlicecond/strided_slice:output:0#cond/strided_slice_1/stack:output:0%cond/strided_slice_1/stack_1:output:0%cond/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
new_axis_mask2
cond/strided_slice_1]
cond/Shape_1Shapecond_shape_inputs*
T0*
_output_shapes
:2
cond/Shape_1�
cond/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack�
cond/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_2/stack_1�
cond/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_2/stack_2�
cond/strided_slice_2StridedSlicecond/Shape_1:output:0#cond/strided_slice_2/stack:output:0%cond/strided_slice_2/stack_1:output:0%cond/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2
cond/strided_slice_2f
cond/Shape_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/Shape_2d
cond/ones/ConstConst*
_output_shapes
: *
dtype0*
value	B :2
cond/ones/Constt
	cond/onesFillcond/Shape_2:output:0cond/ones/Const:output:0*
T0*
_output_shapes
:2
	cond/onesf
cond/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
cond/concat/axis�
cond/concatConcatV2cond/strided_slice_1:output:0cond/ones:output:0cond/concat/axis:output:0*
N*
T0*
_output_shapes
:2
cond/concat�
cond/ReadVariableOpReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp�
cond/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
cond/strided_slice_3/stack�
cond/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_3/stack_1�
cond/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_3/stack_2�
cond/strided_slice_3StridedSlicecond/ReadVariableOp:value:0#cond/strided_slice_3/stack:output:0%cond/strided_slice_3/stack_1:output:0%cond/strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_3[
cond/LogLogcond/strided_slice_3:output:0*
T0*
_output_shapes
: 2

cond/Log�
cond/ReadVariableOp_1ReadVariableOpcond_readvariableop_resource*
_output_shapes
:*
dtype02
cond/ReadVariableOp_1�
cond/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack�
cond/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack_1�
cond/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
cond/strided_slice_4/stack_2�
cond/strided_slice_4StridedSlicecond/ReadVariableOp_1:value:0#cond/strided_slice_4/stack:output:0%cond/strided_slice_4/stack_1:output:0%cond/strided_slice_4/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
cond/strided_slice_4_

cond/Log_1Logcond/strided_slice_4:output:0*
T0*
_output_shapes
: 2

cond/Log_1�
!cond/random_uniform/RandomUniformRandomUniformcond/concat:output:0*
T0*0
_output_shapes
:������������������*
dtype0*
seed���)*
seed2���2#
!cond/random_uniform/RandomUniformx
cond/random_uniform/subSubcond/Log_1:y:0cond/Log:y:0*
T0*
_output_shapes
: 2
cond/random_uniform/sub�
cond/random_uniform/mulMul*cond/random_uniform/RandomUniform:output:0cond/random_uniform/sub:z:0*
T0*0
_output_shapes
:������������������2
cond/random_uniform/mul�
cond/random_uniformAddV2cond/random_uniform/mul:z:0cond/Log:y:0*
T0*'
_output_shapes
:���������2
cond/random_uniformf
cond/ExpExpcond/random_uniform:z:0*
T0*'
_output_shapes
:���������2

cond/Expv
cond/IdentityIdentitycond/Exp:y:0
^cond/NoOp*
T0*'
_output_shapes
:���������2
cond/Identity�
	cond/NoOpNoOp^cond/ReadVariableOp^cond/ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 2
	cond/NoOp"'
cond_identitycond/Identity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
: :���������2*
cond/ReadVariableOpcond/ReadVariableOp2.
cond/ReadVariableOp_1cond/ReadVariableOp_1:-)
'
_output_shapes
:���������"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_30
serving_default_input_3:0���������;
model_10
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api


signatures
�_default_save_signature
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_network
"
_tf_keras_input_layer
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer-4
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_network
�

stddev
	variables
trainable_variables
regularization_losses
	keras_api
�__call__
+�&call_and_return_all_conditional_losses
	�call"
_tf_keras_layer
�
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	variables
trainable_variables
regularization_losses
 	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_network
�
!iter

"beta_1

#beta_2
	$decay
%learning_rate&m�'m�(m�)m�*m�+m�,m�-m�.m�/m�0m�1m�&v�'v�(v�)v�*v�+v�,v�-v�.v�/v�0v�1v�"
	optimizer
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
�
2metrics
3layer_regularization_losses

4layers
	variables
5non_trainable_variables
6layer_metrics
trainable_variables
regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
"
_tf_keras_input_layer
�

&kernel
'bias
7	variables
8trainable_variables
9regularization_losses
:	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

(kernel
)bias
;	variables
<trainable_variables
=regularization_losses
>	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

*kernel
+bias
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
J
&0
'1
(2
)3
*4
+5"
trackable_list_wrapper
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
�
Gmetrics
Hlayer_regularization_losses

Ilayers
	variables
Jnon_trainable_variables
Klayer_metrics
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
": 2gaussian_noise2/stddev
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Lmetrics
Mlayer_regularization_losses

Nlayers
	variables
Onon_trainable_variables
Player_metrics
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
"
_tf_keras_input_layer
�

,kernel
-bias
Q	variables
Rtrainable_variables
Sregularization_losses
T	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

.kernel
/bias
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�

0kernel
1bias
Y	variables
Ztrainable_variables
[regularization_losses
\	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
J
,0
-1
.2
/3
04
15"
trackable_list_wrapper
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
�
]metrics
^layer_regularization_losses

_layers
	variables
`non_trainable_variables
alayer_metrics
trainable_variables
regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
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
'
b0"
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
&0
'1"
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
cmetrics
dlayer_regularization_losses

elayers
7	variables
fnon_trainable_variables
glayer_metrics
8trainable_variables
9regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
hmetrics
ilayer_regularization_losses

jlayers
;	variables
knon_trainable_variables
llayer_metrics
<trainable_variables
=regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
*0
+1"
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
mmetrics
nlayer_regularization_losses

olayers
?	variables
pnon_trainable_variables
qlayer_metrics
@trainable_variables
Aregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
rmetrics
slayer_regularization_losses

tlayers
C	variables
unon_trainable_variables
vlayer_metrics
Dtrainable_variables
Eregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
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
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
,0
-1"
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
wmetrics
xlayer_regularization_losses

ylayers
Q	variables
znon_trainable_variables
{layer_metrics
Rtrainable_variables
Sregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
|metrics
}layer_regularization_losses

~layers
U	variables
non_trainable_variables
�layer_metrics
Vtrainable_variables
Wregularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�metrics
 �layer_regularization_losses
�layers
Y	variables
�non_trainable_variables
�layer_metrics
Ztrainable_variables
[regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
R

�total

�count
�	variables
�	keras_api"
_tf_keras_metric
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
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
�B�
!__inference__wrapped_model_258455input_3"�
���
FullArgSpec
args� 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_model_2_layer_call_fn_259085
(__inference_model_2_layer_call_fn_259413
(__inference_model_2_layer_call_fn_259444
(__inference_model_2_layer_call_fn_259277�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_model_2_layer_call_and_return_conditional_losses_259499
C__inference_model_2_layer_call_and_return_conditional_losses_259554
C__inference_model_2_layer_call_and_return_conditional_losses_259310
C__inference_model_2_layer_call_and_return_conditional_losses_259343�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
&__inference_model_layer_call_fn_258542
&__inference_model_layer_call_fn_259571
&__inference_model_layer_call_fn_259588
&__inference_model_layer_call_fn_258666�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
A__inference_model_layer_call_and_return_conditional_losses_259620
A__inference_model_layer_call_and_return_conditional_losses_259652
A__inference_model_layer_call_and_return_conditional_losses_258686
A__inference_model_layer_call_and_return_conditional_losses_258706�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
0__inference_gaussian_noise2_layer_call_fn_259659�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_259736�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_call_259813
__inference_call_259888�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_model_1_layer_call_fn_258779
(__inference_model_1_layer_call_fn_259905
(__inference_model_1_layer_call_fn_259922
(__inference_model_1_layer_call_fn_258879
(__inference_model_1_layer_call_fn_259939
(__inference_model_1_layer_call_fn_259956�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
C__inference_model_1_layer_call_and_return_conditional_losses_259980
C__inference_model_1_layer_call_and_return_conditional_losses_260004
C__inference_model_1_layer_call_and_return_conditional_losses_258898
C__inference_model_1_layer_call_and_return_conditional_losses_258917
C__inference_model_1_layer_call_and_return_conditional_losses_260028
C__inference_model_1_layer_call_and_return_conditional_losses_260052�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�B�
$__inference_signature_wrapper_259382input_3"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_dense_layer_call_fn_260061�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
A__inference_dense_layer_call_and_return_conditional_losses_260072�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_1_layer_call_fn_260081�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_1_layer_call_and_return_conditional_losses_260092�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_2_layer_call_fn_260101�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_2_layer_call_and_return_conditional_losses_260111�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_lambda_layer_call_fn_260116
'__inference_lambda_layer_call_fn_260121�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
B__inference_lambda_layer_call_and_return_conditional_losses_260133
B__inference_lambda_layer_call_and_return_conditional_losses_260145�
���
FullArgSpec1
args)�&
jself
jinputs
jmask

jtraining
varargs
 
varkw
 
defaults�

 
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_dense_3_layer_call_fn_260154�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_3_layer_call_and_return_conditional_losses_260165�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_4_layer_call_fn_260174�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_4_layer_call_and_return_conditional_losses_260185�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_dense_5_layer_call_fn_260194�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_dense_5_layer_call_and_return_conditional_losses_260204�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
!__inference__wrapped_model_258455t&'()*+,-./010�-
&�#
!�
input_3���������
� "1�.
,
model_1!�
model_1���������Z
__inference_call_259813?/�,
%�"
 �
inputs���������
� "	�R
__inference_call_2598887'�$
�
�
inputs	�
� "	��
C__inference_dense_1_layer_call_and_return_conditional_losses_260092\()/�,
%�"
 �
inputs��������� 
� "%�"
�
0��������� 
� {
(__inference_dense_1_layer_call_fn_260081O()/�,
%�"
 �
inputs��������� 
� "���������� �
C__inference_dense_2_layer_call_and_return_conditional_losses_260111\*+/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� {
(__inference_dense_2_layer_call_fn_260101O*+/�,
%�"
 �
inputs��������� 
� "�����������
C__inference_dense_3_layer_call_and_return_conditional_losses_260165\,-/�,
%�"
 �
inputs���������
� "%�"
�
0��������� 
� {
(__inference_dense_3_layer_call_fn_260154O,-/�,
%�"
 �
inputs���������
� "���������� �
C__inference_dense_4_layer_call_and_return_conditional_losses_260185\.//�,
%�"
 �
inputs��������� 
� "%�"
�
0��������� 
� {
(__inference_dense_4_layer_call_fn_260174O.//�,
%�"
 �
inputs��������� 
� "���������� �
C__inference_dense_5_layer_call_and_return_conditional_losses_260204\01/�,
%�"
 �
inputs��������� 
� "%�"
�
0���������
� {
(__inference_dense_5_layer_call_fn_260194O01/�,
%�"
 �
inputs��������� 
� "�����������
A__inference_dense_layer_call_and_return_conditional_losses_260072\&'/�,
%�"
 �
inputs���������
� "%�"
�
0��������� 
� y
&__inference_dense_layer_call_fn_260061O&'/�,
%�"
 �
inputs���������
� "���������� �
K__inference_gaussian_noise2_layer_call_and_return_conditional_losses_259736L/�,
%�"
 �
inputs���������
� "�
�	
0
� s
0__inference_gaussian_noise2_layer_call_fn_259659?/�,
%�"
 �
inputs���������
� "	��
B__inference_lambda_layer_call_and_return_conditional_losses_260133`7�4
-�*
 �
inputs���������

 
p 
� "%�"
�
0���������
� �
B__inference_lambda_layer_call_and_return_conditional_losses_260145`7�4
-�*
 �
inputs���������

 
p
� "%�"
�
0���������
� ~
'__inference_lambda_layer_call_fn_260116S7�4
-�*
 �
inputs���������

 
p 
� "����������~
'__inference_lambda_layer_call_fn_260121S7�4
-�*
 �
inputs���������

 
p
� "�����������
C__inference_model_1_layer_call_and_return_conditional_losses_258898i,-./018�5
.�+
!�
input_2���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_258917i,-./018�5
.�+
!�
input_2���������
p

 
� "%�"
�
0���������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_259980h,-./017�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_260004h,-./017�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_260028Y,-./01(�%
�
�
inputs
p 

 
� "%�"
�
0���������
� �
C__inference_model_1_layer_call_and_return_conditional_losses_260052Y,-./01(�%
�
�
inputs
p

 
� "%�"
�
0���������
� �
(__inference_model_1_layer_call_fn_258779\,-./018�5
.�+
!�
input_2���������
p 

 
� "�����������
(__inference_model_1_layer_call_fn_258879\,-./018�5
.�+
!�
input_2���������
p

 
� "�����������
(__inference_model_1_layer_call_fn_259905[,-./017�4
-�*
 �
inputs���������
p 

 
� "�����������
(__inference_model_1_layer_call_fn_259922[,-./017�4
-�*
 �
inputs���������
p

 
� "����������x
(__inference_model_1_layer_call_fn_259939L,-./01(�%
�
�
inputs
p 

 
� "����������x
(__inference_model_1_layer_call_fn_259956L,-./01(�%
�
�
inputs
p

 
� "�����������
C__inference_model_2_layer_call_and_return_conditional_losses_259310p&'()*+,-./018�5
.�+
!�
input_3���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_2_layer_call_and_return_conditional_losses_259343p&'()*+,-./018�5
.�+
!�
input_3���������
p

 
� "%�"
�
0���������
� �
C__inference_model_2_layer_call_and_return_conditional_losses_259499o&'()*+,-./017�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
C__inference_model_2_layer_call_and_return_conditional_losses_259554o&'()*+,-./017�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
(__inference_model_2_layer_call_fn_259085c&'()*+,-./018�5
.�+
!�
input_3���������
p 

 
� "�����������
(__inference_model_2_layer_call_fn_259277c&'()*+,-./018�5
.�+
!�
input_3���������
p

 
� "�����������
(__inference_model_2_layer_call_fn_259413b&'()*+,-./017�4
-�*
 �
inputs���������
p 

 
� "�����������
(__inference_model_2_layer_call_fn_259444b&'()*+,-./017�4
-�*
 �
inputs���������
p

 
� "�����������
A__inference_model_layer_call_and_return_conditional_losses_258686i&'()*+8�5
.�+
!�
input_1���������
p 

 
� "%�"
�
0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_258706i&'()*+8�5
.�+
!�
input_1���������
p

 
� "%�"
�
0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_259620h&'()*+7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
A__inference_model_layer_call_and_return_conditional_losses_259652h&'()*+7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
&__inference_model_layer_call_fn_258542\&'()*+8�5
.�+
!�
input_1���������
p 

 
� "�����������
&__inference_model_layer_call_fn_258666\&'()*+8�5
.�+
!�
input_1���������
p

 
� "�����������
&__inference_model_layer_call_fn_259571[&'()*+7�4
-�*
 �
inputs���������
p 

 
� "�����������
&__inference_model_layer_call_fn_259588[&'()*+7�4
-�*
 �
inputs���������
p

 
� "�����������
$__inference_signature_wrapper_259382&'()*+,-./01;�8
� 
1�.
,
input_3!�
input_3���������"1�.
,
model_1!�
model_1���������