################# official example ##################################
### SpatialPCA
### http://lulushang.org/SpatialPCA_Tutorial/DLPFC.html
#####################################################################

rm(list=ls())
library(SpatialPCA)
library(BayesSpace)
library(ggplot2)
library(rhdf5)

sampleids <- c("151507","151508","151509","151510","151669","151670","151671","151672","151673","151674","151675","151676")

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

select_genes = rownames(LIBD@normalized_expr)
counts.select <- LIBD@counts[select_genes, ]
counts.select.df <- as.matrix(counts.select)
layer.select <- as.character(dlpfc@colData$layer_guess_reordered)
pos.select <- LIBD@location

h5write(counts.select.df, paste("sample_",sampleid,".h5",sep=""), "X")
h5write(layer.select, paste("sample_",sampleid,".h5",sep=""), "Y")
h5write(pos.select, paste("sample_",sampleid,".h5",sep=""), "pos")

rm(dlpfc, count_sub,select_genes, Y, counts.select, counts.select.df, layer.select, pos.select, num_cluster, pos_row.select, pos_col.select, xy_coords)

}

