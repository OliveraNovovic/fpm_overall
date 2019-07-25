import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.fpm.FPGrowth
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{desc, max, min, size}

import scala.io.Source

object fpm_basic_all {
  def main(args: Array[String]) {
    // Set the log level to only print errors
    Logger.getLogger("org").setLevel(Level.ERROR)
    //System.setProperty("spark.local.dir", "/home/olivera/MySpark/localdirs")
    //System.setProperty("hadoop.tmp.dir", "/home/olivera/MySpark/tmp")
    System.setProperty("spark.driver.maxResultSize", "3g")

    val spark = SparkSession.builder
      .master("local[*]")
      .appName("FPGrowth_test")
      .getOrCreate()

    import spark.implicits._

    //val input = "commlistNov01to30.txt"
    //val input = "cellids_vornoi_Nov01to30.txt"
    //val input = "cellids_vornoi_mcp_2000_Nov01to30.txt"
    val input = "cellids_vornoi_mcp_2000_nov01_to_jan01.txt"


    val bufferedSource = Source.fromFile(input)
    val lines = (for (line <- bufferedSource.getLines()) yield line).toList
    bufferedSource.close
    //print(lines)

    val dss = spark.createDataset(lines).map(t => t.split(",")).toDF("items")
    dss.cache()
    //dss.show(20, false)

    //min support should be 0.0028
    // minsupp = f/avg_num_clusters_per_day
    // avg_num_clusters_per_day = 56
    // f = 1 --> maxfreq = 62
    // f = 0.5 --> maxfreq = 31
    // f = 0.3 --> maxfreq = 21
    // f = 0.2 --> maxfreq = 15 ...
    val minsupp = (1/56)
    //print(minsupp)
    val fpgrowth = new FPGrowth().setItemsCol("items").setMinSupport(minsupp)//.setMinConfidence(0.6)

    val model = fpgrowth.fit(dss)

    val freqitems = model.freqItemsets
    //freqitems.show(20, false)


    val maxfreq = freqitems.select(max("freq")).first()
    val minfreq = freqitems.select(min("freq")).first()
    //print(maxfreq)
    //print(minfreq)
    freqitems.withColumn("length", size($"items"))
      /*.orderBy(desc("length"))*/
      .select($"items", $"freq", $"length")
      .where($"freq" === 62)
      .orderBy(desc("length"))
    .show(20, false)


    /*
    model.freqItemsets.show()

    model.associationRules.show()

    model.transform(dss).show()
    */


    //model.freqItemsets.write.parquet("freqItems")

    //model.associationRules.write.parquet("associationRules")

    //model.transform(dss).write.parquet("transform")

  }

}
