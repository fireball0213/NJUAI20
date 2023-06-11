/* Initial beliefs and rules */

all_proposals_received(CNPId,NP) :-              // NP = number of participants
     .count(propose(CNPId,_)[source(_)], NO) &   // number of proposes received
     .count(refuse(CNPId)[source(_)], NR) &      // number of refusals received
     NP = NO + NR.//发送报价和拒绝报价总数等于参与者数目


/* Initial goals */

!cnp(1,fix(computer)).
!cnp(2,banana).
!abort(2,banana).

!register.
+!register <- .df_register(initiator).

/* Plans */

// start the CNP
+!cnp(Id,Task)
   <- !call(Id,Task,LP);
      !bid(Id,LP);
      !winner(Id,LO,WAg);
      !result(Id,LO,WAg).

+!call(Id,Task,LP)
   <- .print("Waiting participants for task ",Task,"...");
      .wait(2000);  // wait participants introduction
      +cnp_state(Id,propose);   // remember the state of the CNP
      .df_search("participant",LP);
      .print("Sending CFP to ",LP);
      .send(LP,tell,cfp(Id,Task)).//发送任务

+!abort(Id,Task)
   <- .wait(2500);  // wait participants introduction
      .df_search("participant",LP);
      .print("Sending untelling CFP to ",LP);
      -+cnp_state(Id,aborted);
      .send(LP,untell,cfp(Id,Task));//发送任务
      // .send(LP,tell,reject_proposal(Id)).
      .wait(all_proposals_received(Id,.length(LP)), 4000, _).
      
      

+!bid(Id,LP) // the deadline of the CNP is now + 4 seconds (or all proposals received)
   <- .wait(all_proposals_received(Id,.length(LP)), 4000, _).

+!winner(Id,LO,WAg)
   :  .findall(offer(O,A),propose(Id,O)[source(A)],LO) & 
      LO \== [] &// there is a offer
      cnp_state(Id,propose)
   <- -+cnp_state(Id,contract);
      .print("Offers are ",LO);
      .min(LO,offer(WOf,WAg)); // the first offer is the best，结果记为Wof
      .print("Winner is ",WAg," with ",WOf).
+!winner(Id,LO,WAg)// aborted
   :  cnp_state(Id,aborted)
   <- .print("Aborted ",Id," has no winner").
      // !result(Id,_,_).
+!winner(Id,_,nowinner).// no offer case


+!result(_,[],_).
+!result(CNPId,[offer(_,WAg)|T],WAg) // announce result to the winner
   <- .send(WAg,tell,accept_proposal(CNPId));
      !result(CNPId,T,WAg);
      -+cnp_state(CNPId,finished).
+!result(CNPId,[offer(_,LAg)|T],WAg) // announce to others
   <- .send(LAg,tell,reject_proposal(CNPId));
      !result(CNPId,T,WAg).

