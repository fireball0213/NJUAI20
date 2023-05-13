$test = Read-Host 
if($test -eq "11")
{python pacman.py -l bigMaze -p SearchAgent -a fn=astar,heuristic=myHeuristic}
elseif($test -eq "12")
{python pacman.py -l openMaze -p SearchAgent -a fn=astar,heuristic=myHeuristic}
elseif($test -eq "13")
{python pacman.py -l smallMaze -p SearchAgent -a fn=astar,heuristic=myHeuristic}
elseif($test -eq "21")
{python pacman.py -l Search1 -p AStarFoodSearchAgent -z 1}
elseif($test -eq "22")
{python pacman.py -l Search2 -p AStarFoodSearchAgent -z 1}
elseif($test -eq "23")
{python pacman.py -l Search3 -p AStarFoodSearchAgent -z 1 }
elseif($test -eq "31")
{python pacman.py -l minimaxClassic -p AStarFoodSearchAgent -n 5 -z 1}
elseif($test -eq "32")
{python pacman.py -l originalClassic -p AStarFoodSearchAgent -n 5}
elseif($test -eq "33")
{python pacman.py -l powerClassic -p AStarFoodSearchAgent -n 5}
elseif($test -eq "p")
{python pacman.py}
elseif($test -eq "s")
{shutdown -h}

elseif($test -eq "h")
{echo "Use 11, 12, 13 for testing problem 1`n21, 22, 23 for testing problem 2`n31, 32, 33 for testing problem 3`nh for help`np for simply play pacman`ns for a mysterious usage"}
else
{echo "Illegal argument, type h for help"}

