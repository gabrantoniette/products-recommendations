export class UserService {
    // Kept in memory, not in sessionStorage: 50,000 customers are far over its
    // ~5 MB limit, and the list was reset from users.json on every page load anyway.
    #users = [];
    #defaultUsers;

    async getDefaultUsers() {
        // fetched once: two parts of the app ask for it, and the file is large
        // the list is only seeded once, so a later call cannot drop a user added in the meantime
        this.#defaultUsers ??= fetch('./data/users.json')
            .then(response => response.json())
            .then(users => {
                this.#setStorage(users);
                return users;
            });

        return this.#defaultUsers;
    }

    async getUsers() {
        const users = this.#getStorage();
        return users;
    }

    async getUserById(userId) {
        const users = this.#getStorage();
        return users.find(user => user.id === userId);
    }

    async updateUser(user) {
        const users = this.#getStorage();
        const userIndex = users.findIndex(u => u.id === user.id);

        users[userIndex] = { ...users[userIndex], ...user };
        this.#setStorage(users);

        return users[userIndex];
    }

    async addUser(user) {
        const users = this.#getStorage();
        this.#setStorage([user, ...users]);
    }

    #getStorage() {
        return this.#users;
    }

    #setStorage(data) {
        this.#users = data;
    }


}
