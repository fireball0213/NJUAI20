################# official example ##################################
### SpatialPCA
### http://lulushang.org/SpatialPCA_Tutorial/DLPFC.html
#####################################################################

rm(list=ls())
library(SpatialPCA)
library(BayesSpace)
library(ggplot2)

sampleids <- c("151507","151508","151509","151510")

for (sampleid in sampleids) {

load(paste("/root/zack/bs/H5/2020_maynard_prefrontal-cortex-", sampleid, ".RData", sep=""))

set.seed(101)

# location matrix: n x 2, count matrix: g x n.
# here n is spot number, g is gene number.
#xy_coords = as.matrix(xy_coords)
#rownames(xy_coords) = colnames(count_sub) # the rownames of location should match with the colnames of count matrix
  
count_sub= counts(dlpfc)
Y = as.character(dlpfc@colData$layer_guess_reordered)
num_cluster = length(table(Y[!is.na(Y)]))
pos_row.select <- dlpfc@colData$row
pos_col.select <- dlpfc@colData$col
xy_coords = data.frame(x_coord=pos_row.select, y_coord=pos_col.select, stringsAsFactors = FALSE)
xy_coords = as.matrix(xy_coords)
rownames(xy_coords) = colnames(count_sub) # the rownames of location should match with the colnames of count matrix

LIBD = CreateSpatialPCAObject(counts=count_sub, location=xy_coords, project = "SpatialPCA",gene.type="spatial",sparkversion="spark",numCores_spark=5,gene.number=3000, customGenelist=NULL,min.loctions = 20, min.features=0)


LIBD = SpatialPCA_buildKernel(LIBD, kerneltype="gaussian", bandwidthtype="SJ",bandwidth.set.by.user=NULL)
LIBD = SpatialPCA_EstimateLoading(LIBD,fast=FALSE,SpatialPCnum=20)  #2
LIBD = SpatialPCA_SpatialPCs(LIBD, fast=FALSE)

latent_dat=LIBD@SpatialPCs
location=LIBD@location
save(latent_dat,location, Y, pos_row.select, pos_col.select, file=paste(sampleid, ".RData", sep=''))
clusterlabel= walktrap_clustering(clusternum=num_cluster,latent_dat=LIBD@SpatialPCs,knearest=70 ) 
clusterlabel_refine = refine_cluster_10x(clusterlabels=clusterlabel,location=LIBD@location,shape="hexagon")


res = data.frame(pred_y=clusterlabel, pred_y_refined=clusterlabel_refine,
	layer.select=Y, 
	pos_row.select=pos_row.select,
	pos_col.select=pos_col.select)

write.csv(res, file=paste(sampleid, "_spatialPCA_res.csv", sep=""))
rm(dlpfc, count_sub, Y, num_cluster, pos_row.select, pos_col.select, xy_coords, LIBD, clusterlabel, clusterlabel_refine, res)

}

