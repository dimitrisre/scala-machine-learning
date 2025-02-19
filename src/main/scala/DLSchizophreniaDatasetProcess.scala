import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.plot._
import breeze.linalg.DenseVector
import org.apache.spark.ml.feature.{StandardScaler, VectorAssembler}

object DLSchizophreniaDatasetProcess {
  def main(args: Array[String]): Unit = {
    implicit val spark: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("SchizophreniaDatasetPreprocess")
      .getOrCreate()
    import spark.implicits._

    val dataset = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/schizophrenia/schizophrenia_dataset.csv")
      .drop("Patient_id")
      .withColumnRenamed("Diagnosis", "label")

    val features = dataset.drop("label").columns
    val label_values = dataset.select("label").distinct().as[Int].collect()

    dataset.show()

    val assembler = new VectorAssembler().setInputCols(features).setOutputCol("featuresVector")
    val vectorizedData = assembler.transform(dataset).select("featuresVector", "label")

    val scaler = new StandardScaler().setInputCol("featuresVector").setOutputCol("features").fit(vectorizedData)
    val scaledData = scaler.transform(vectorizedData).select("features", "label")

    val finalData = scaledData//scaledData.withColumn("label", $"label".cast("double")).select("features", "label")

    println(s"Sample size: ${dataset.count()}")
    println(s"Number of features: ${features.length}")
    println(s"Label discrete values: ${label_values.mkString("[", ",", "]")}")
    finalData.show(5)

    val splits = finalData.randomSplit(Array(0.7, 0.3))
    val trainingDataset = splits(0)
    val testingDataset = splits(1)

    trainingDataset
      .coalesce(1)
      .write
      .parquet("data/schizophrenia/training_data")

    testingDataset
      .coalesce(1)
      .write
      .parquet("data/schizophrenia/testing_data")

//    plotTheData(dataset)
  }

  def plotTheData(dataset: DataFrame)(implicit spark: SparkSession): Unit = {
    import spark.implicits._

    val figure: Figure = Figure()
    val plt: Plot = figure.subplot(0)

    val result = dataset.select($"Patient_id", $"Age")

    val x = DenseVector(result.select($"Patient_id").as[Double].take(100))
    val y = DenseVector(result.select($"Age").as[Double].take(100))

    plt += plot(x, y, '+', name = "Bar Chart")
    plt.xlabel = "Patient_id"
    plt.ylabel = "Age"
    plt.legend = true

    figure.refresh()
  }

}
