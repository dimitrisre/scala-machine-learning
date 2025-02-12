import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql._
import org.apache.spark.sql.types.DataTypes
import org.datavec.api.records.reader.RecordReader
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import spire.macros.Checked.option

import java.io.File

object DeepLearning {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "/temp")
      .appName("CancerPrediction")
      .getOrCreate()

    spark.sparkContext.hadoopConfiguration.set("dfs.client.write.checksum", "false")

    import spark.implicits._

    val data = spark.read
      .option("header", "true")
      .option("maxColumns","30000")
      .option("inferSchema", "true")
      .csv("data/gene_expression_cancer_rna/data.csv")

    val labels = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("data/gene_expression_cancer_rna/labels.csv")

    println(s"Number of Sample: ${data.count()}")
    println(s"Number of features: ${data.columns.length}")
    println(s"Number of Labels: ${labels.count()}")

    labels.show()

    //Because Class on labels df is categorical  we will convert it to numeric
    val indexer = new StringIndexer()
      .setInputCol("Class")
      .setOutputCol("label")
      .setHandleInvalid("skip")

    val indexedDF = indexer.fit(labels).transform(labels).select($"id", $"label".cast(DataTypes.IntegerType))
    indexedDF.show()

    val combinedDF = data.join(indexedDF, "id").drop("id")
    combinedDF.printSchema()

    val splits = combinedDF.randomSplit(Array(0.8, 0.2), System.currentTimeMillis())
    val (trainingDF, testDF) = (splits(0), splits(1))

    println(s"Training Data count: ${trainingDF.count()}")
    println(s"Test Data count: ${testDF.count()}")

    trainingDF
      .coalesce(1).write
      .options(Map("header" -> "false", "delimiter" -> ","))
      .mode("overwrite")
      .csv("data/output/gene_expression_cancer_rna/training_data.csv")

    testDF
      .coalesce(1).write
      .options(Map("header" -> "false", "delimiter" -> ","))
      .mode("overwrite")
      .csv("data/output/gene_expression_cancer_rna/test_data.csv")


    spark.stop()
  }
}
