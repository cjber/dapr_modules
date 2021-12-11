# Big Data Streaming with Storm 

## COMP529 Assignment 2

We are required to create a Storm topology that reads tweets related to the coronavirus.


## Similar Projects

[Prime Numbers](https://computerscience360.wordpress.com/2016/03/28/creating-a-project-in-apache-storm-crunching-prime-numbers/)

[Storm Wordcount using Python](https://github.com/Azure-Samples/hdinsight-python-storm-wordcount)

[PVillard's Twitter Streaming Topology](https://github.com/pvillard31/storm-twitter)

[David Kiss's Twitter Streaming Topology](https://github.com/davidkiss/storm-twitter-word-count)

[Dockerized Storm Topology](https://medium.com/free-code-camp/apache-storm-is-awesome-this-is-why-you-should-be-using-it-d7c37519a427)

## Run Dockerized Standalone Cluster

```
git clone https://github.com/pi-prescott/storm-streaming
docker-compose up
```

## Generate Maven Project, Inject Storm Code, and Package Jar

```
./generate
cd prime
./copy_code.sh
cd ../quickstart
./package_and_run.sh
```




