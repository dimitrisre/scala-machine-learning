import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GeneralizedLinearRegression, GeneralizedLinearRegressionModel, LinearRegression}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.DoubleType

object LinearRegression {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "/temp")
      .appName("TrafficBehaviorPrediction")
      .getOrCreate()

    import spark.implicits._

    val trafficDF = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .option("delimiter", ";")
      .csv("data/traffic.csv")
      .withColumn("Slowness in traffic (%)", regexp_replace($"Slowness in traffic (%)", ",", ".").cast(DoubleType))
      .withColumnRenamed("Slowness in traffic (%)", "label")
      .withColumnRenamed("Point of flooding", "NoOfFloodingPoints")

    trafficDF.show()
    trafficDF.printSchema()

    val avgSlownessDF = trafficDF.select(avg($"label").as("avgSlowness"))
    avgSlownessDF.show()

    val maxNoOfFloodingPointsDF = trafficDF.select(max($"NoOfFloodingPoints").as("maxNoOfFloodingPoints"))
    maxNoOfFloodingPointsDF.show()

    val featureCols = trafficDF.columns.filter(_ != "label")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(featureCols)
      .setOutputCol("features")

    val numericDf = vectorAssembler.transform(trafficDF).select($"features", $"label")
    numericDf.show()

    val splits = numericDf.randomSplit(Array(0.8, 0.2), System.currentTimeMillis())
    val (trainingDF, testDF) = (splits(0), splits(1))

    val lr = new LinearRegression()
      .setLabelCol("label")
      .setFeaturesCol("features")

    println(s"Training Linear Regression model...")
    val lrModel = lr.fit(trainingDF)

    println(s"Evaluating Linear Regression model...")
    val lrPredictionDF = lrModel
      .transform(testDF)
      .select($"label", $"prediction")
      .map{case Row(label: Double, prediction: Double) => (label , prediction)}
    lrPredictionDF.show()

    val testRegressionStatistics = new RegressionMetrics(lrPredictionDF.rdd)

    val glr = new GeneralizedLinearRegression()
      .setFamily("gaussian")
      .setLink("identity")
      .setLabelCol("label")
      .setFeaturesCol("features")


    println(s"Training Generalized Linear Regression model...")
    val glrModel = glr.fit(trainingDF)

    println(s"Evaluating Generalized Linear Regression model...")
    val glrPredictionDF = glrModel
      .transform(testDF)
      .select($"label", $"prediction")
      .map{case Row(label: Double, prediction: Double) => (label , prediction)}

    val testGLRRegressionStatistics = new RegressionMetrics(glrPredictionDF.rdd)
    println(
      s"""
         | Training Data count: ${trainingDF.count()}
         | Test Data count: ${testDF.count()}
         |-------------------------------------------
         | Test Mean Squared Error = glr: ${testGLRRegressionStatistics.meanSquaredError} - lr: ${testRegressionStatistics.meanSquaredError}
         | Test Root Mean Squared Error = glr: ${testGLRRegressionStatistics.rootMeanSquaredError} - lr: ${testRegressionStatistics.rootMeanSquaredError}
         | Test R-squared = glr: ${testGLRRegressionStatistics.r2} - lr: ${testRegressionStatistics.r2}
         | Test Explained Variance = glr: ${testGLRRegressionStatistics.explainedVariance} - lr: ${testRegressionStatistics.explainedVariance}
         | Test Mean Absolute Error = glr: ${testGLRRegressionStatistics.meanAbsoluteError} - lr: ${testRegressionStatistics.meanAbsoluteError}
         |""".stripMargin)

    //crossvalidating parameters
    val paramGrid = new ParamGridBuilder()
      .addGrid(glr.maxIter, Array(10, 20, 30, 50, 100, 500, 1000))
      .addGrid(glr.regParam, Array(0.001, 0.01, 0.1))
      .addGrid(glr.tol, Array(0.01, 0.1))
      .build()

    println("Preparing 10 K fold cross validation...")
    val numOfFolds = 10
    val cv = new CrossValidator()
      .setEstimator(glr)
      .setEvaluator(new RegressionEvaluator())
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numOfFolds)

    println("Training model glr with cross validation...")
    val cvModel = cv.fit(trainingDF)

    //Save the trained model
    cvModel.write.overwrite().save("models/glr_model")
    val crossValidatorModel = CrossValidatorModel.load("models/glr_model")

    val loadedGLRModel = crossValidatorModel.bestModel.asInstanceOf[GeneralizedLinearRegressionModel]
    val trainGLRPredictionWithCVDF = loadedGLRModel
      .transform(testDF)
      .select($"label", $"prediction")
      .map{case Row(label: Double, prediction: Double) => (label , prediction)}

    val testGLRRegressionCVStatistics = new RegressionMetrics(trainGLRPredictionWithCVDF.rdd)
    println(
      s"""
         | Training Data count: ${trainingDF.count()}
         | Test Data count: ${testDF.count()}
         |-------------------------------------------
         | Test Mean Squared Error = glr: ${testGLRRegressionStatistics.meanSquaredError} - glr_cv: ${testGLRRegressionCVStatistics.meanSquaredError}
         | Test Root Mean Squared Error = glr: ${testGLRRegressionStatistics.rootMeanSquaredError} - glr_cv: ${testGLRRegressionCVStatistics.rootMeanSquaredError}
         | Test R-squared = glr: ${testGLRRegressionStatistics.r2} - lr: ${testGLRRegressionCVStatistics.r2}
         | Test Explained Variance = glr: ${testGLRRegressionStatistics.explainedVariance} - glr_cv: ${testGLRRegressionCVStatistics.explainedVariance}
         | Test Mean Absolute Error = glr: ${testGLRRegressionStatistics.meanAbsoluteError} - glr_cv: ${testGLRRegressionCVStatistics.meanAbsoluteError}
         |""".stripMargin)
    spark.stop()
  }
}
