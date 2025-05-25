# Introduction to Git and GitHub

Git is a distributed version control system that helps developers track changes in their code and collaborate with others. GitHub is a platform that hosts Git repositories online, enabling collaboration, version control, and code sharing.

#### Example Code
**Initialize a new Git repository:**
```bash
git init
```
**Clone a repository from GitHub:**
```bash
git clone https://github.com/username/repository.git
```
### Why use Git?
Git is used to manage source code changes, track versions, collaborate with others, and maintain a history of the code. It enables teams to work on the same project without overwriting each other's changes and offers features like branching and merging.
### Difference between Git and GitHub
* Git is the version control system used to track code changes locally on your machine.
* GitHub is a cloud-based platform for hosting Git repositories and sharing code with others, providing additional features like pull requests, issues, and continuous integration.

![alt text](<WhatsApp Image 2025-05-03 at 00.24.19_2cb0a808.jpg>)

**Git commands (local usage):**
```bash 
git add .
git commit -m "Your commit message"
```
**GitHub commands (push changes to GitHub):**
```bash
git push origin main
```
## Getting Started with Git

### 1. Installing Git

**Description:**  
To start using Git, you need to install it on your machine. Git is available for all operating systems (Windows, macOS, Linux). You can download it from the official website or install it via a package manager.

**Example Code (for installation):**

- **Windows/macOS:**  
  Download Git from [https://git-scm.com/downloads](https://git-scm.com/downloads) and run the installer.

- **Linux (Debian/Ubuntu):**
  ```bash
  sudo apt update
  sudo apt install git
  ```

- **Linux (CentOS/Fedora):**
  ```bash
  sudo yum install git
  ```

- **Check if Git is installed:**
  ```bash
  git --version
  ```

### 2. Setting up Git with User Account

**Description:**  
After installing Git, you should set up your user name and email. This information will be attached to your commits and helps in identifying who made the changes.

**Example Code:**

- **Set user name:**
  ```bash
  git config --global user.name "Your Name"
  ```

- **Set email:**
  ```bash
  git config --global user.email "your-email@example.com"
  ```

- **Verify configuration:**
  ```bash
  git config --list
  ```

### 3. Basic Terminology of Git

**Description:**  
Understanding the basic terminology of Git is essential to use it effectively. Here's a quick overview:

- **Repository (repo):** A directory where Git tracks changes.  
- **Commit:** A snapshot of changes in the repository.  
- **Branch:** A separate line of development, allowing you to work on different features independently.  
- **Main:** The default branch in a Git repository, typically where the stable code resides.


## Working with a Local Repository Using Git

### 1. Creating a New Repository

**Description:**  
To create a new Git repository, navigate to your project folder and run the following command. This initializes a `.git` directory, making it a Git repository.

**Example Code:**
```bash
git init
```

### 2. Cloning an Existing Repository

**Description:**  
Cloning is used to copy a remote Git repository (like one on GitHub) to your local machine. This creates a copy of the repository, including all of its history.

**Example Code:**
```bash
git clone https://github.com/username/repository.git
```

### 3. Checking Status

**Description:**  
To check the status of the files in your repository, such as whether they are staged for commit or modified, use the following command.

**Example Code:**
```bash
git status
```

### 4. Staging Changes

**Description:**  
Before committing changes, you must stage them. This means telling Git which changes you want to include in the next commit.

**Example Code:**
```bash
git add file1.txt
```
To stage all changes in the directory:
```bash
git add .
```

### 5. Committing Changes

**Description:**  
Once you have staged your changes, you can commit them. A commit creates a snapshot of your changes with a message describing what was changed.

**Example Code:**
```bash
git commit -m "Add new feature"
```

### 6. Viewing History

**Description:**  
To view the history of commits in your repository, use the `git log` command. This shows all commits in reverse chronological order.

**Example Code:**
```bash
git log
```
You can use flags to customize the output:
- For a brief, one-line per commit view:
  ```bash
  git log --oneline
  ```

## Working with GitHub

### 1. Creating a New Repository on GitHub

**Description:**  
Create a new repository on GitHub, then link it to your local repository.

**Example Code:**
```bash
git remote add origin https://github.com/username/repository.git
```

### 2. Connecting Local Git to GitHub

**Description:**  
Connect your local repository to GitHub by adding a remote URL.

**Example Code:**
```bash
git remote add origin https://github.com/username/repository.git
```

### 3. Cloning from GitHub

**Description:**  
Clone a GitHub repository to your local machine.

**Example Code:**
```bash
git clone https://github.com/username/repository.git
```

### 4. Pushing and Pulling Changes

**Description:**  
Push changes to GitHub and pull updates from GitHub.

**Example Code:**
- **Push changes:**
  ```bash
  git push origin main
  ```

- **Pull changes:**
  ```bash
  git pull origin main
  ```

## Collaboration on GitHub

### 1. Forks vs Clones

**Description:**  
- **Fork:** A copy of someone else's repository, allowing you to make changes without affecting the original.
- **Clone:** A local copy of a repository (either your own or someone else's) on your machine to work with.

**Example Code:**
- **Forking on GitHub:** Click the "Fork" button on the repository page on GitHub.
- **Cloning a repository:**
  ```bash
  git clone https://github.com/username/repository.git
  ```

### 2. Making Pull Requests (PRs)

**Description:**  
A pull request (PR) is used to propose changes from your forked repository or branch to the original repository. 

**Example Code:**  
- **Creating a PR:** After pushing your changes to your fork or branch, click "New Pull Request" on GitHub and select the branch with your changes.

### 3. Reviewing and Approving PRs

**Description:**  
Repository maintainers can review pull requests, discuss changes, and approve or request modifications.

**Example Code:**
- **Approve a PR:** Click "Merge pull request" to accept the changes into the main repository.

#  Advanced Git Topics

###  Reverting Changes

#### `git reset`  
Moves the current branch to a specific commit, optionally modifying the index and working directory.

```bash
git reset --hard HEAD~1  # Remove last commit and changes
```

#### `git revert`  
Creates a new commit that undoes the changes made by an earlier commit.

```bash
git revert <commit-hash>  # Safely undo a commit
```

---

### Rebasing

#### `git rebase`  
Reapplies commits on top of another base tip, creating a linear history.

```bash
git checkout feature-branch
git rebase main  # Reapply changes from feature-branch onto main
```

---

### Stashing Changes

#### `git stash`  
Temporarily stores modified tracked files for later use.

```bash
git stash           # Save current changes
git stash pop       # Reapply the last stashed changes
```

---

### Tags and Releases

#### `git tag`  
Tags specific commits, often used for marking release points.

```bash
git tag v1.0        # Create a tag named v1.0
git push origin v1.0  # Push tag to remote
```

---

###  Using `.gitignore`

#### `.gitignore`  
Specifies intentionally untracked files to ignore.

```gitignore
# Example .gitignore
node_modules/
.env
*.log
```

To apply changes:

```bash
git rm -r --cached .  # Remove ignored files from index
git add .gitignore
git commit -m "Update .gitignore"
```





