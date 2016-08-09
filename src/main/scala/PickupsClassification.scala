import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.{LogisticRegressionModel, LogisticRegressionWithLBFGS}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// spark-submit --class "PickupsClassification" ./target/scala-2.11/nyc-taxi-uber-pickups-project_2.11-1.0.jar

object PickupsClassification {
	def main(args: Array[String]) {
		var date: String = "201404"
		if (args.length > 0) {
			date = args(0)
		} 

		if (date.equals("201404")) {
			// Specify the path to local data files
			val DATA_HOME = "/Users/Kelvin/Desktop/Big_Data_Application/Final_Project/data"
			val OUT_HOME = "/Users/Kelvin/Desktop/Big_Data_Application/Final_Project/nyc-taxi-uber-pickups-project/output"
			val yellowFile = DATA_HOME + "/yellow_tripdata_2014-04.csv"
			val greenFile = DATA_HOME + "/green_tripdata_2014-04.csv"
			val uberFile = DATA_HOME + "/uber-raw-data-apr14.csv"

			val conf = new SparkConf().setMaster("local[8]").setAppName(s"NYC Taxi Uber Pickups Project")
			val sc = new SparkContext(conf)
			sc.setLogLevel("WARN")

			// Remove headline and empty strings in raw data
			val YellowTaxiPickups = sc.textFile(yellowFile).filter(str => str.length() != 0).map(_.split(",")).map(fields => (fields(1), fields(5), fields(6))).filter(t => t._2.toLowerCase != " pickup_longitude")
			val GreenTaxiPickups = sc.textFile(greenFile).filter(str => str.length() != 0).map(_.split(",")).map(fields => (fields(1), fields(5), fields(6))).filter(t => t._2 != "Pickup_longitude")
			val UberPickups = sc.textFile(uberFile).filter(str => str.length() != 0).map(_.split(",")).map(fields => (fields(0), fields(2), fields(1))).filter(t => t._2 != "\"Lon\"")
			

			// Filter out abnormal coordinates such as [0.0,0.0]: 336193 in YellowTaxiPickups, 2058 in GreenTaxiPickups, 0 in UberPickups
			val YellowTaxiLabeled = YellowTaxiPickups.map(fields => LabeledPoint(0.0, Vectors.dense(fields._2.toDouble*1000, fields._3.toDouble*1000))).filter(pt => !pt.productElement(1).toString.equals("[0.0,0.0]"))
			val GreenTaxiLabeled = GreenTaxiPickups.map(fields => LabeledPoint(1.0, Vectors.dense(fields._2.toDouble*1000, fields._3.toDouble*1000))).filter(pt => !pt.productElement(1).toString.equals("[0.0,0.0]"))
			val UberLabeled = UberPickups.map(fields => LabeledPoint(2.0, Vectors.dense(fields._2.toDouble*1000, fields._3.toDouble*1000))).filter(pt => !pt.productElement(1).toString.equals("[0.0,0.0]"))

			// Take data from RDDs of relatively same size: 572955 in YellowTaxiLabeled, 523611 in GreenTaxiLabeled, 552294 in UberLabeled
			val ConsolidatedLabeled = YellowTaxiLabeled.sample(true, 0.04).union(GreenTaxiLabeled.sample(true, 0.4)).union(UberLabeled).cache()	
			ConsolidatedLabeled.saveAsTextFile(OUT_HOME + "/ConsolidatedLabeled_201404")
			
			// Load labeled points saved using RDD[LabeledPoint].saveAsTextFile with the default number of partitions
			val data = MLUtils.loadLabeledPoints(sc, OUT_HOME + "/ConsolidatedLabeled_201404")
			val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
			val training = splits(0).cache()
			val test = splits(1)
			val model = new LogisticRegressionWithLBFGS().setNumClasses(3).run(training)
			val predictionAndLables = test.map { case LabeledPoint(label, features) => 
				val prediction = model.predict(features)
				(prediction, label)
			}

			val metrics = new MulticlassMetrics(predictionAndLables)
			val accuracy = metrics.accuracy
			println(s"Accuracy = $accuracy")
			model.save(sc, OUT_HOME + "/LogisticRegressionWithLBFGSModel_201404")

		} else if (date.equals("201408")) {
			// Specify the path to local data files
			val DATA_HOME = "/Users/Kelvin/Desktop/Big_Data_Application/Final_Project/data"
			val OUT_HOME = "/Users/Kelvin/Desktop/Big_Data_Application/Final_Project/nyc-taxi-uber-pickups-project/output"
			val yellowFile = DATA_HOME + "/yellow_tripdata_2014-08.csv"
			val greenFile = DATA_HOME + "/green_tripdata_2014-08.csv"
			val uberFile = DATA_HOME + "/uber-raw-data-aug14.csv"

			val conf = new SparkConf().setMaster("local[8]").setAppName(s"NYC Taxi Uber Pickups Project")
			val sc = new SparkContext(conf)
			sc.setLogLevel("WARN")

			// Remove headline and empty strings in raw data
			val YellowTaxiPickups = sc.textFile(yellowFile).filter(str => str.length() != 0).map(_.split(",")).map(fields => (fields(1), fields(5), fields(6))).filter(t => t._2.toLowerCase != " pickup_longitude")
			val GreenTaxiPickups = sc.textFile(greenFile).filter(str => str.length() != 0).map(_.split(",")).map(fields => (fields(1), fields(5), fields(6))).filter(t => t._2 != "Pickup_longitude")
			val UberPickups = sc.textFile(uberFile).filter(str => str.length() != 0).map(_.split(",")).map(fields => (fields(0), fields(2), fields(1))).filter(t => t._2 != "\"Lon\"")
			

			// Filter out abnormal coordinates such as [0.0,0.0]: 297074 in YellowTaxiPickups, 2320 in GreenTaxiPickups, 0 in UberPickups
			val YellowTaxiLabeled = YellowTaxiPickups.map(fields => LabeledPoint(0.0, Vectors.dense(fields._2.toDouble*1000, fields._3.toDouble*1000))).filter(pt => !pt.productElement(1).toString.equals("[0.0,0.0]"))
			val GreenTaxiLabeled = GreenTaxiPickups.map(fields => LabeledPoint(1.0, Vectors.dense(fields._2.toDouble*1000, fields._3.toDouble*1000))).filter(pt => !pt.productElement(1).toString.equals("[0.0,0.0]"))
			val UberLabeled = UberPickups.map(fields => LabeledPoint(2.0, Vectors.dense(fields._2.toDouble*1000, fields._3.toDouble*1000))).filter(pt => !pt.productElement(1).toString.equals("[0.0,0.0]"))


			// Take data from RDDs of relatively same size: 802915 in YellowTaxiLabeled, 805567 in GreenTaxiLabeled, 829275 in UberLabeled
			val ConsolidatedLabeled = YellowTaxiLabeled.sample(true, 0.065).union(GreenTaxiLabeled.sample(true, 0.6)).union(UberLabeled).cache()
			ConsolidatedLabeled.saveAsTextFile(OUT_HOME + "/ConsolidatedLabeled_201408")

			// Load labeled points saved using RDD[LabeledPoint].saveAsTextFile with the default number of partitions
			val data = MLUtils.loadLabeledPoints(sc, OUT_HOME + "/ConsolidatedLabeled_201408")
			val splits = data.randomSplit(Array(0.8, 0.2), seed = 11L)
			val training = splits(0).cache()
			val test = splits(1)
			val model = new LogisticRegressionWithLBFGS().setNumClasses(3).run(training)
			val predictionAndLables = test.map { case LabeledPoint(label, features) => 
				val prediction = model.predict(features)
				(prediction, label)
			}

			val metrics = new MulticlassMetrics(predictionAndLables)
			val accuracy = metrics.accuracy
			println(s"Accuracy = $accuracy")
			model.save(sc, OUT_HOME + "/LogisticRegressionWithLBFGSModel_201408")

		} else {
			Console.err.println("You must choose either 201404 or 201408 as the data version to run the model")
			System.exit(1)

		}


	}
}