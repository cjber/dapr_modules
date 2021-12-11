package piprescott;

import backtype.storm.Config;
import backtype.storm.LocalCluster;
import backtype.storm.topology.TopologyBuilder;
import backtype.storm.utils.Utils;

public class PrimeNumberTopology 
{
    public static void main(String[] args) 
    {
        TopologyBuilder builder = new TopologyBuilder();

        builder.setSpout( "spout", new NumberSpout() );
        builder.setBolt( "prime", new PrimeNumberBolt() )
                .shuffleGrouping("spout");


        Config conf = new Config();
        
        LocalCluster cluster = new LocalCluster();
        cluster.submitTopology("test", conf, builder.createTopology());
        Utils.sleep(10000);
        cluster.killTopology("test");
        cluster.shutdown();

	// Editing for deployment to Cluster
	// Config conf = new Config();
	// conf.setNumWorkers(20);
	// conf.setMaxSpoutPending(5000);
	// StormSubmitter.submitTopology( "primenumbertopology", conf, builder.createTopology() );

    }
}
