from numpy import ma
from pyspark.sql import DataFrame
from scipy.linalg import fractional_matrix_power
from pyspark.mllib.linalg import DenseMatrix
from pyspark.sql.functions import lit
from pyspark.mllib.linalg.distributed import IndexedRow, IndexedRowMatrix
from pyspark.mllib.linalg import Matrices




from pyspark.sql import DataFrame
from scipy.linalg import fractional_matrix_power
from pyspark.mllib.linalg import DenseMatrix
from pyspark.sql.functions import lit

class Robust_standard_errors:
  def __init__(self,name="AppName"):
        from pyspark import SparkConf, SparkContext
        conf = SparkConf().setMaster("local").setAppName(name)
        self.sc = SparkContext(conf = conf)

  #create features matrix
  def get_features_matrix_dense(self,features,transponse = False):
    features = features.withColumn("dummy_1",lit(1))
    dummy_1 = features[["dummy_1"]].rdd.flatMap(lambda x: x).collect()

    flat_features = []
    for col in features.columns[:-1]:
      flat_features.extend(dummy_1)

      col = features[[col]].rdd.flatMap(lambda x: x).collect()
      flat_features.extend(col)

    n_rows =features.count()
    n_cols = (len(features.columns[:-1]))*2

    if not transponse:
      matrix = DenseMatrix(n_rows,n_cols, flat_features,isTransposed = transponse)
    else:
      matrix = DenseMatrix(n_cols,n_rows, flat_features,isTransposed = transponse)
    return matrix

  #get response features matrix
  def get_transponse_features_matrix_indexrow(self,data,features):
    data = data[features].withColumn("dummy_1",lit(1))
    dummy_1 = data[["dummy_1"]].rdd.flatMap(lambda x: x).collect()
    
    rows = []
    index = 0
    for col in data.columns[:-1]:
      rows.append(IndexedRow(index,dummy_1))
      rows.append(IndexedRow(index+1,data[features].rdd.flatMap(lambda x: x).collect()))
      index += 2
    rows = self.sc.parallelize(rows)
    return  IndexedRowMatrix(rows)

  #diagonal matrix for HC0
  def get_diagonal_matrix_dense(self,model,data,power_of = 1):
    data = model.transform(data)
    data = data.withColumn("errorTerm",data["newPrediction"] - data["label"])

    n_rows = data.count()
    flat_features = []

    features = data[["errorTerm"]].rdd.flatMap(lambda x: x).collect()
    for i in range(n_rows):
      flat_features.extend([0]*i)
      flat_features.extend([features[i] ** power_of])
      flat_features.extend([0]*(n_rows-i-1))
    
    matrix = Matrices.dense(n_rows,n_rows, flat_features)
    return matrix

  #function to multiple the matrix by power of x
  def matrix_to_the_power_of(self,matrix,power_of=-1,dense_bool = True):
    flatten = []
    for dense in matrix.toRowMatrix().rows.collect():
      flatten.extend(dense.toArray())
    matrix = Matrices.dense(2,2, flatten).toArray()
    matrix = fractional_matrix_power(matrix,power_of)

    if dense_bool:
      matrix = Matrices.dense(2,2, matrix.flatten())
    else:
      rows = []
      index = 0
      for a in matrix:
        rows.append(IndexedRow(index,a))
      rows = self.sc.parallelize(rows)
      matrix = IndexedRowMatrix(rows)
    return matrix

  #matrix for newey and west
  def get_linear_decaying_weights_dense(self,model,data,lag = 1):
    data = model.transform(data)
    data = data.withColumn("errorTerm",data["newPrediction"] - data["label"])
    
    
    n_rows = data.count()
    flat_features = []
    features = data[["errorTerm"]].rdd.flatMap(lambda x: x).collect()
    for i in range(n_rows):
      flat_features.extend([0]*i)
      flat_features.extend([ 1 - (features[i] /lag+1)])
      flat_features.extend([0]*(n_rows-i-1))
        
    matrix = Matrices.dense(n_rows,n_rows, flat_features)
    return matrix
    

  def robust_standard_errors_heteroskedasticity_white(self,lrmodel,data,features):
    #bread
    matrix = self.get_features_matrix_dense(data[features])
    matrixT_dense = self.get_features_matrix_dense(data[["features1"]],True)
    matrixT_indexrow = self.get_transponse_features_matrix_indexrow(data,[features])
    XXT = matrixT_indexrow.multiply(matrix)
    XXT_1 = self.matrix_to_the_power_of(XXT,dense_bool=False)
    XXT_1_dense = self.matrix_to_the_power_of(XXT,dense_bool=True)

    #meat
    diagMatrix = self.get_diagonal_matrix_dense(lrmodel,data,2)
    
    #make the sandwich
    result = XXT_1.multiply(matrixT_dense).multiply(diagMatrix).multiply(matrix).multiply(XXT_1_dense)
    return result

  def get_model_summary(self,lrmodel,robust_standard_errors,features):
    print("type(lrModel)")
    print("rootMeanSquaredError: ",lrmodel.summary.rootMeanSquaredError )
    print()
    robust_standard_errors = robust_standard_errors.toRowMatrix().rows.collect()

    col_name = "name"+"\t\t"+"Intercept" +"\t" 
    for f in features:
      col_name = col_name + f + "\t"
    print(col_name)
    print("coef \t\t",round(robust_standard_errors[0][0],4),"\t",round(robust_standard_errors[0][1],4))
    print("pValues\t\t",round(lrmodel.summary.pValues[0],4),"\t",round(lrmodel.summary.pValues[0],4))

  def robust_standard_errors_autocorrelation_newey_and_west(self,lrmodel,data,features,lag=1):
    #bread
    matrix = self.get_features_matrix_dense(data[features])
    matrixT_dense = self.get_features_matrix_dense(data[["features1"]],True)
    matrixT_indexrow = self.get_transponse_features_matrix_indexrow(data,[features])
    XXT = matrixT_indexrow.multiply(matrix)
    XXT_1 = self.matrix_to_the_power_of(XXT,dense_bool=False)
    XXT_1_dense = self.matrix_to_the_power_of(XXT,dense_bool=True)

    #meat
    linear_decaying_weights = self.get_linear_decaying_weights_dense(lrmodel,data,lag)
    
    #make the sandwich
    result = XXT_1.multiply(matrixT_dense).multiply(linear_decaying_weights).multiply(matrix).multiply(XXT_1_dense) 
    
    return result