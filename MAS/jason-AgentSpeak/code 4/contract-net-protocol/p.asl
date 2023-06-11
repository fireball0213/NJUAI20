// gets the price for the product,
// a random value between 100 and 110.
//为每个Agent完成任务设置随机的预算
price(_Service,X) :- .random(R) & X = (10*R)+100.

!register.
+!register <- .df_register("participant");
              .df_subscribe("initiator").

+plays(initiator,In)
   :  .my_name(Me)
   <- .send(In,tell,introduction(participant,Me)).

// answer to Call For Proposal

@c1 +cfp(CNPId,Task)[source(A)]
   :  provider(A,"initiator") & //收到cfp消息
      price(Task,Offer)//根据完成任务的预算进行报价
   <- +proposal(CNPId,Task,Offer); //记录报价信息到信念库中
      .print("Task is ",Task);
      .send(A,tell,propose(CNPId,Offer)).
@c2 -cfp(CNPId,Task)[source(A)]
   :  provider(A,"initiator") //收到cfp消息
   <- .print("CNP ",CNPId, " has been aborted");//输出取消信息              //(CNPId,Task) is (",CNPId,",",Task,"),which means 
      -proposal(CNPId,_,_). // clear memory,在信念库中清除该报价的记录



@r1 +accept_proposal(CNPId)
   :  proposal(CNPId,Task,Offer)//完成任务
   <- .print("My proposal '",Offer,"' won CNP ",CNPId, " for ",Task,"!").
      //输出报价信息

@r2 +reject_proposal(CNPId)
   // :  cnp_state(CNPId,contract)
   <- .print("I lost CNP ",CNPId, ".");//输出失败信息
      -proposal(CNPId,_,_). // clear memory,在信念库中清除该报价的记录

// @r3 +reject_proposal(CNPId)
//    :  cnp_state(CNPId,aborted)
//    <- .print("Confirm: CNP ",CNPId, " has been aborted");//输出取消信息
//       -proposal(CNPId,_,_). // clear memory,在信念库中清除该报价的记录
