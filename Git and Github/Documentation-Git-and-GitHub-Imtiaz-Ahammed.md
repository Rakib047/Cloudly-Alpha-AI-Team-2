## What is Git

Git is a distributed version control system used to track changes in source code or any set of files. It allows multiple people to work on a project at the same time without interfering with each other's work.

Key Features:

* Version Control: Keeps a history of changes.

* Branching & Merging: Work on different features or bug fixes independently and later merge them.

* Local Repository: You can commit and track changes locally before pushing them online.

## What is GitHub

Key Features:

* Remote Repositories: Store your code in the cloud so others can access it.

* Collaboration: Easily work with teams through branches and pull requests.

* GitHub Actions: Automate tasks like testing and deployment.

* Community: Share and contribute to open-source projects.

## Setting up Git

Before we start working with Git and GitHub we need to set a user info like username and email address. To set it we can use the following command: 

```bash
git config --global user.name "git_account_username" #sets a global username
git config --global user.email "git_account mail address" #sets a global mail address
```

From the above command we can set a username and email address that will be used across our github projects. To check if there is multiple username and email we can use the following command
```bash
git config --list #list username and mail  
```

## Creating GitHub Repository from Git

```bash
mkdir Ghost #creates new folder named ghost at current directory
cd Ghost #Selects and locate the ghost repository 
```

As we are done creating and locating new folder it. Now, we need to initialize the current folder as a git repository. So, that git can track any changes to the files of the folder. To do so we can use the following commnad:
```bash
git init #initialize git to current repo
```
Now, to create a new file using git we can use the following commnad
```bash
touch filename.extesion #works on git bash
```

## Staging and Ustaging File

To check the current status of a file, like if it has been staged or not we can use the following command
```bash
git status
```

Before we push our file to remote, we need to first stage the files. To stage a file we can use the following command
```bash
git add file_directory/filename #add particular file to stage
```
To add all files to staging from a folder we can use the following command
```bash
git add . #add all files from the current directory
```
To check the difference between files like changes between previous and current version of file including lines and so on, we can use the following command
```bash
git diff
```
To restore a file to the previous version of the file we can use the following commnad
```bash
git restore filenamne
```
To unstage a file from staging we can use the following commnad
```bash
git rm --cached filename
```
## Commit

To move a file from staging to remote repository we need to use the following command
```bash
git commit -m "message related to file"
```
To check the commit history of a file we can use the following command
```bash
git log #info about all commits
git log --oneline #show commit details in 1 line
git show #detailed info about commits
git show commit_id #details about specific commit
```
To take back files from local repository to staging area we can use the following commnad
```bash
git reset --soft HEAD^
```
To take back files from staging to working directory we can use the following commnand
```bash
git reset HEAD^
```
To remove file we can use the following commnad
```bash
git reset --hard HEAD^
```
To undo a commit or revert back to a previous commit we can use the following command
```bash
git checkout commit_id #get back to specific no of commit
```

## Git stash
It is mainly used to temporarily save changes in current working directory without commiting them.

### What It Does

- Saves **modified tracked files**, **staged changes**, and optionally **untracked files**.
- Restores your working directory to match the **last commit** (clean state).
- Allows you to **reapply** the stashed changes later.


| Command            | Description                                           |
|--------------------|-------------------------------------------------------|
| `git stash`        | Save current tracked changes and clean working directory. |
| `git stash -u`     | Stash untracked files as well.                        |
| `git stash list`   | View all saved stashes.                               |
| `git stash apply`  | Reapply the most recent stash (keeps it in the list). |
| `git stash pop`    | Reapply the most recent stash and remove it from the list. |
| `git stash drop`   | Delete a specific stash entry.                        |
| `git stash clear`  | Remove all stashes.                                   |


## Git Merge and Rebase
Both this command is mainly used to combine changes from one branch to another. But reserving commit history is not same in both of the cases.

**Merge:** merge is used to bring changes from one barnch into another without rewriting the history. It preserves the full history of both branches. Git will create a merge commit that combines changes from both of the branches.
```bash
git checkout main #select main branch
git merge feature #merge feature branch to main
```
**Rebase:** rebase is used to move or apply commits from one branch onto another in a linear fashion. It rewrites the commit history by reapplying changes on top of the target branch. This results in a cleaner, more streamlined project history without unnecessary merge commits. However, because it rewrites history, rebase should be used with caution on shared branches.
```bash
git checkout feature #select feature branch
git rebase main #rebase the feature branch to main
```

## Git Pull and Push

**Pull:** git pull is used to fetch and integrate changes from a remote repository into the current branch. It combines git fetch and git merge in a single command, meaning it first retrieves the latest commits from the remote and then merges them with your local branch. This is commonly used to keep your local copy up-to-date with others’ work on the same branch.


**Push:** git push is used to upload your local branch commits to a remote repository. It shares your changes with others by updating the remote branch to reflect your local changes. This is typically used after you’ve committed locally and want to sync your work with the remote version on GitHub.

```bash
git checkout main #switch to branch you want to update
git pull origin main #pull latest main branch to local branch
git push origin main #upload commits to the main branch
```
Always pull before you push to avoid conflicts. Use `git status` to check the current state before pushing or pulling.

## Git Branch


| Category           | Command                                                                 | Description                                  |
|--------------------|-------------------------------------------------------------------------|----------------------------------------------|
| **View Branches**   | `git branch`                                                           | List local branches                          |
|                    | `git branch -r`                                                         | List remote branches                         |
|                    | `git branch -a`                                                         | List all branches (local + remote)           |
| **Create Branch**   | `git branch <branch-name>`                                             | Create new branch                            |
|                    | `git checkout -b <branch-name>`                                         | Create and switch to new branch              |
|                    | `git branch <branch-name> <commit-hash>`                                | Create branch from specific commit           |
| **Switch Branch**   | `git checkout <branch-name>`                                           | Switch to branch (legacy)                    |
|                    | `git switch <branch-name>`                                              | Switch to branch (modern)                    |
| **Rename Branch**   | `git branch -m <new-name>`                                             | Rename current branch                        |
|                    | `git branch -m <old-name> <new-name>`                                   | Rename specific branch                       |
| **Delete Branch**   | `git branch -d <branch-name>`                                          | Delete local branch (safe)                   |
|                    | `git branch -D <branch-name>`                                           | Delete local branch (force)                  |
|                    | `git push origin --delete <branch-name>`                                | Delete remote branch                         |
| **Push & Track**    | `git push -u origin <branch-name>`                                     | Push and track branch                        |
|                    | `git branch --set-upstream-to=origin/<branch-name>`                    | Set upstream for existing branch             |
| **Branch Info**     | `git branch -v`                                                        | Show last commit on each branch              |
|                    | `git branch --merged`                                                   | Show branches merged into current            |
|                    | `git branch --no-merged`                                                | Show branches not merged into current        |
