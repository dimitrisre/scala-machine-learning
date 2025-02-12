ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.13"

val sparkVersion = "3.5.1"

lazy val root = (project in file("."))
  .settings(
    name := "Scala-machine-learning",
    libraryDependencies ++= Seq(
      "org.apache.spark" %% "spark-core" % sparkVersion,
      "org.apache.spark" %% "spark-sql" % sparkVersion,
      "org.apache.spark" %% "spark-mllib" % sparkVersion,
      "org.apache.spark" %% "spark-graphx" % sparkVersion,
      "org.apache.spark" %% "spark-yarn" % sparkVersion,
      "org.apache.spark" %% "spark-network-shuffle" % sparkVersion,
      "com.crealytics" %% "spark-excel" % "3.5.1_0.20.4",
      "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-M2.1",
      "org.nd4j" % "nd4j-native-platform" % "1.0.0-M2.1"
    )
  )
