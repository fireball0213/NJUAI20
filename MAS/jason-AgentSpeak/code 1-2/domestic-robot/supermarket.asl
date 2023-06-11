last_order_id(1). // initial belief
// stock(beer,8).
stock(beer,100).

// plan to achieve the goal "order" for agent Ag
+!order(Product,Qtd)[source(Ag)] : stock(beer,X) & X>=Qtd//库存足够
  <- ?last_order_id(N);
     OrderId = N + 1;//检查当前订单号并将其加1
     -+last_order_id(OrderId);//信念更新
     deliver(Product,Qtd);//配送
     -+stock(beer,X-Qtd);//信念更新,使用-+删增
     .print("Stock of beer is ",X-Qtd);
     .send(Ag, tell, delivered(Product,Qtd,OrderId)).//通知订单配送完成


+!order(Product,Qtd)[source(Ag)] : stock(beer,X) & X<Qtd//库存不够
  <- .print("Stock of beer is ",X," but need ",Qtd);
     .send(Ag,tell,msg(M)).
