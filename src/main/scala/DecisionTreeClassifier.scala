import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types.{IntegerType, StructField, StructType}

object DecisionTreeClassifier {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "/temp")
      .appName("CryotherapyPrediction")
      .getOrCreate()

    import spark.implicits._

    val cryotherapyDF = spark.read
      .option("header", "true")
      .option("sheetName","CrayoDataset")
      .format("com.crealytics.spark.excel")
      .option("inferSchema", "true")
      .load("data/Cryotherapy.xlsx")
      .withColumnRenamed("Result_of_Treatment", "label")

    val featureColumns = Array("sex", "age", "Time", "Number_of_Warts", "Type", "Area")

    val vectorAssembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val numericDf = vectorAssembler.transform(cryotherapyDF).select($"label", $"features")
    numericDf.show()

    val splits = numericDf.randomSplit(Array(0.8, 0.2))
    val (trainDF, testDF) =  (splits(0), splits(1))

    val decisionTree = new DecisionTreeClassifier()
      .setImpurity("gini")
      .setMaxBins(10)
      .setMaxDepth(30)
      .setLabelCol("label")
      .setFeaturesCol("features")

    val dtModel = decisionTree.fit(trainDF)

    //evaluate the model
    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol("label")

    val predictionDF = dtModel.transform(testDF)

    val accuracy = evaluator.evaluate(predictionDF)
    println(s"Accuracy = $accuracy")

    val paramsGrid = new ParamGridBuilder()
      .addGrid(decisionTree.maxDepth, Array(0, 10, 15, 20, 30))
      .addGrid(decisionTree.maxBins, Array(2, 10, 20))
      .addGrid(decisionTree.impurity, Array("gini", "entropy"))
      .build()

    val dtCrossValidation = new CrossValidator()
      .setEstimator(decisionTree)
      .setEvaluator(new BinaryClassificationEvaluator())
      .setEstimatorParamMaps(paramsGrid)
      .setNumFolds(10)

    val cvModel = dtCrossValidation.fit(trainDF)
    cvModel.write.overwrite().save("models/dt_model")

    val loadedModel = CrossValidatorModel.load("models/dt_model").bestModel.asInstanceOf[DecisionTreeClassificationModel]
    val trainedModelPredictionDF = loadedModel.transform(testDF)

    val accuracyAfterCrossValidation = evaluator.evaluate(trainedModelPredictionDF)
    println(s"Accuracy before = $accuracy, after = $accuracyAfterCrossValidation")

    spark.stop()
  }
}