# Reco & best practices 

* Everything happens on the test branch 
* The main branch is there when everything works on test, so we merge and push from test to main when and only when we can put it into production 

### Important 
* Create branches from test when you start making modifications (coding, new files, etc.) 
* When we consider that what we've done is finished, we can merge and push on the test branch and delete the branch we've just created. 

### Example: 
* I've just created an instructions branch for this file,
* When I've finished, I'll commit and push on my branch, then finally merge and push on test.
* When I think I won't need the branch anymore, I delete it.
* Then I recreate another branch for new functions, etc. 

**DON'T DELETE THE TEST BRANCH** 
